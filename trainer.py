from __future__ import annotations

import argparse
import copy
import hashlib
from enum import IntEnum
from typing import Union, Optional, Dict

import gymnasium as gym
import numpy as np
import ray
from gymnasium import spaces
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Lava, Wall, Door
from minigrid.envs import EmptyEnv, MultiRoom
from minigrid.wrappers import RGBImgObsWrapper
from ray import tune
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module import RLModule
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType, PolicyID
from ray.tune import register_env, CheckpointConfig

from intrinsic_motivation.dowham_v2 import DoWhaMIntrinsicRewardV2


class CustomCallback(RLlibCallback):
    def on_episode_start(
            self,
            *,
            episode: Union[EpisodeType, EpisodeV2],
            env_runner: Optional["EnvRunner"] = None,
            metrics_logger: Optional[MetricsLogger] = None,
            env: Optional[gym.Env] = None,
            env_index: int,
            rl_module: Optional[RLModule] = None,
            # TODO (sven): Deprecate these args.
            worker: Optional["EnvRunner"] = None,
            base_env: Optional[BaseEnv] = None,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            **kwargs,
    ) -> None:
        episode.custom_metrics["left"] = 0
        episode.custom_metrics["right"] = 0
        episode.custom_metrics["forward"] = 0
        episode.custom_metrics["pickup"] = 0
        episode.custom_metrics["drop"] = 0
        episode.custom_metrics["toggle"] = 0
        episode.custom_metrics["done"] = 0

    def on_episode_step(
            self,
            *,
            episode: Union[EpisodeType, EpisodeV2],
            env_runner: Optional["EnvRunner"] = None,
            metrics_logger: Optional[MetricsLogger] = None,
            env: Optional[gym.Env] = None,
            env_index: int,
            rl_module: Optional[RLModule] = None,
            # TODO (sven): Deprecate these args.
            worker: Optional["EnvRunner"] = None,
            base_env: Optional[BaseEnv] = None,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            **kwargs,
    ) -> None:
        env = base_env.get_sub_environments()[env_index].unwrapped
        episode.custom_metrics["intrinsic_reward"] = env.intrinsic_reward
        episode.custom_metrics["step_done"] = env.done
        episode.custom_metrics[env.actions(env.action).name] += 1

    def on_episode_created(
            self,
            *,
            episode: Union[EpisodeType, EpisodeV2],
            env_runner: Optional["EnvRunner"] = None,
            metrics_logger: Optional[MetricsLogger] = None,
            env: Optional[gym.Env] = None,
            env_index: int,
            rl_module: Optional[RLModule] = None,
            # TODO (sven): Deprecate these args.
            worker: Optional["EnvRunner"] = None,
            base_env: Optional[BaseEnv] = None,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            **kwargs,
    ) -> None:
        env = base_env.get_sub_environments()[env_index].unwrapped
        episode.custom_metrics["percentage_visited"] = env.percentage_visited


class CustomEnv(EmptyEnv):
    class Environments(IntEnum):
        empty = 0
        crossing = 1
        four_rooms = 2
        multi_room = 3

    class NavigationOnlyActions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        done = 6

    def __init__(self, **kwargs):
        print(f"Environment Config: {kwargs}")
        self.env_type = kwargs.pop("env_type", CustomEnv.Environments.multi_room)
        self.enable_dowham_reward_v1 = kwargs.pop('enable_dowham_reward_v1', False)
        self.enable_dowham_reward_v2 = kwargs.pop('enable_dowham_reward_v2', False)
        print(f"Enable Dowham Reward V2: {self.enable_dowham_reward_v2}")

        self.percentage_visited = 0.0
        self.action = None
        self.reward_range = (0, 1)
        self.max_steps = 200
        self.dowham_reward = None
        self.tile_size = 24
        self.highlight = False

        super().__init__(
            size=11,
            tile_size=self.tile_size,
            highlight=self.highlight,
            **kwargs)

        self.states = np.full((self.width, self.height), 0)

        if self.enable_dowham_reward_v2:
            self.reward_range = (-1, 1)
            self.dowham_reward = DoWhaMIntrinsicRewardV2(eta=40, H=1, tau=0.5)

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(
                self.height * self.tile_size,
                self.width * self.tile_size,
                3,
            ),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {'direction': self.observation_space.spaces['direction'], "image": new_image_space}
        )

        print(f"Environment Type: {CustomEnv.Environments(self.env_type).name}")

        if self.env_type == CustomEnv.Environments.crossing:
            self.obstacle_type = Wall
            self.num_crossings = 1
            # self.actions = CustomEnv.NavigationOnlyActions
            # self.action_space = spaces.Discrete(len(self.actions))

        if self.env_type == CustomEnv.Environments.empty:
            # self.actions = CustomEnv.NavigationOnlyActions
            # self.action_space = spaces.Discrete(len(self.actions))
            pass

        self.intrinsic_reward = 0
        self.done = False

    def step(self, action: int):
        self.states[self.agent_pos[0]][self.agent_pos[1]] += 1
        self.action = action
        current_obs = self.img_observation()
        obs, reward, terminated, truncated, _ = super().step(action)
        next_obs = self.img_observation()

        if self.enable_dowham_reward_v1 or self.enable_dowham_reward_v2:
            self.dowham_reward.update_state_visits(current_obs, next_obs)
            state_changed = current_obs != next_obs
            self.dowham_reward.update_usage(current_obs, action)

            self.dowham_reward.update_effectiveness(
                current_obs,
                action,
                next_obs,
                state_changed
            )

            self.intrinsic_reward = self.dowham_reward.calculate_intrinsic_reward(
                current_obs,
                action,
                next_obs,
                state_changed
            )
            self.intrinsic_reward *= 0.05

        if terminated:
            reward = reward * 10
        else:
            reward += self.intrinsic_reward

        self.done = terminated
        return obs, reward, terminated, truncated, {}

    def _gen_grid(self, width, height):
        if self.env_type == CustomEnv.Environments.crossing:
            self.crossing_env(width, height)
        elif self.env_type == CustomEnv.Environments.empty:
            self.empty_env_random_goal(width, height)
        elif self.env_type == CustomEnv.Environments.four_rooms:
            self.four_rooms(width, height)
        elif self.env_type == CustomEnv.Environments.multi_room:
            self.multi_room(width, height)

    def img_observation(self, size=32):
        rgb_img = self.get_frame(
            highlight=self.highlight, tile_size=self.tile_size
        )

        sample_hash = hashlib.sha256()

        # from PIL import Image
        # img = Image.fromarray(rgb_img)
        # img.save("debug_image.png")

        # Convert the full grid to a list for hashing
        full_grid_list = rgb_img.flatten().tolist()

        to_encode = [full_grid_list]

        # Update the hash with each element
        for item in to_encode:
            sample_hash.update(str(item).encode("utf8"))

        # Return the hashed value
        return sample_hash.hexdigest()[:size]

    def crossing_env(self, width, height):
        import itertools as itt
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)
        self.goal_position = (width - 2, height - 2)

        # Place obstacles (lava or walls)
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        # Lava rivers or walls specified by direction and position in grid
        rivers = [(v, i) for i in range(2, height - 2, 2)]
        rivers += [(h, j) for j in range(2, width - 2, 2)]
        self.np_random.shuffle(rivers)
        rivers = rivers[: self.num_crossings]  # sample random rivers
        rivers_v = sorted(pos for direction, pos in rivers if direction is v)
        rivers_h = sorted(pos for direction, pos in rivers if direction is h)
        obstacle_pos = itt.chain(
            itt.product(range(1, width - 1), rivers_h),
            itt.product(rivers_v, range(1, height - 1)),
        )
        for i, j in obstacle_pos:
            self.put_obj(self.obstacle_type(), i, j)

        # Sample path to goal
        path = [h] * len(rivers_v) + [v] * len(rivers_h)
        self.np_random.shuffle(path)

        # Create openings
        limits_v = [0] + rivers_v + [height - 1]
        limits_h = [0] + rivers_h + [width - 1]
        room_i, room_j = 0, 0
        for direction in path:
            if direction is h:
                i = limits_v[room_i + 1]
                j = self.np_random.choice(
                    range(limits_h[room_j] + 1, limits_h[room_j + 1])
                )
                room_i += 1
            elif direction is v:
                i = self.np_random.choice(
                    range(limits_v[room_i] + 1, limits_v[room_i + 1])
                )
                j = limits_h[room_j + 1]
                room_j += 1
            else:
                assert False
            self.grid.set(i, j, None)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

    def empty_env_random_goal(self, width, height):
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        # Get grid size from environment
        grid_size = self.width  # Assuming width == height

        # Randomly assign a new goal position (excluding (1,1))
        while True:
            self.goal_pos = (np.random.randint(1, grid_size - 2), np.random.randint(1, grid_size - 2))
            if self.goal_pos != (1, 1):  # Ensure it's not the starting position
                self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])
                break

    def four_rooms(self, width, height):
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0
        self.goal_pos = (width - 2, height - 2)
        self._agent_default_pos = self.agent_pos
        self._goal_default_pos = self.goal_pos
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):
            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            # assuming random start direction
            self.agent_dir = self._rand_int(0, 4)
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

    def multi_room(self, width, height):
        self.minNumRooms = 2
        self.maxNumRooms = 2
        self.maxRoomSize = 9
        self.max_steps = self.maxNumRooms * 20

        roomList = []

        # Choose a random number of rooms to generate
        numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms + 1)

        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos = (self._rand_int(0, width - 2), self._rand_int(0, width - 2))

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=4,
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos,
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        prevDoorColor = None

        # For each room
        for idx, room in enumerate(roomList):
            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                doorColors = set(COLOR_NAMES)
                if prevDoorColor:
                    doorColors.remove(prevDoorColor)
                # Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                doorColor = self._rand_elem(sorted(doorColors))

                entryDoor = Door(doorColor)
                self.grid.set(room.entryDoorPos[0], room.entryDoorPos[1], entryDoor)
                prevDoorColor = doorColor

                prevRoom = roomList[idx - 1]
                prevRoom.exitDoorPos = room.entryDoorPos

        # Randomize the starting agent position and direction
        self.place_agent(roomList[0].top, roomList[0].size)

        # Place the final goal in the last room
        self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

        self.mission = "traverse the rooms to get to the goal"

    def _placeRoom(self, numLeft, roomList, minSz, maxSz, entryDoorWall, entryDoorPos):
        # Choose the room size randomly
        sizeX = self._rand_int(minSz, maxSz + 1)
        sizeY = self._rand_int(minSz, maxSz + 1)

        # The first room will be at the door position
        if len(roomList) == 0:
            topX, topY = entryDoorPos
        # Entry on the right
        elif entryDoorWall == 0:
            topX = entryDoorPos[0] - sizeX + 1
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif entryDoorWall == 1:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1] - sizeY + 1
        # Entry wall on the left
        elif entryDoorWall == 2:
            topX = entryDoorPos[0]
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif entryDoorWall == 3:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1]
        else:
            assert False, entryDoorWall

        # If the room is out of the grid, can't place a room here
        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > self.width or topY + sizeY >= self.height:
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in roomList[:-1]:
            nonOverlap = (
                    topX + sizeX < room.top[0]
                    or room.top[0] + room.size[0] <= topX
                    or topY + sizeY < room.top[1]
                    or room.top[1] + room.size[1] <= topY
            )

            if not nonOverlap:
                return False

        # Add this room to the list
        roomList.append(MultiRoom((topX, topY), (sizeX, sizeY), entryDoorPos, None))

        # If this was the last room, stop
        if numLeft == 1:
            return True

        # Try placing the next room
        for i in range(0, 8):
            # Pick which wall to place the out door on
            wallSet = {0, 1, 2, 3}
            wallSet.remove(entryDoorWall)
            exitDoorWall = self._rand_elem(sorted(wallSet))
            nextEntryWall = (exitDoorWall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            if exitDoorWall == 0:
                exitDoorPos = (topX + sizeX - 1, topY + self._rand_int(1, sizeY - 1))
            # Exit on south wall
            elif exitDoorWall == 1:
                exitDoorPos = (topX + self._rand_int(1, sizeX - 1), topY + sizeY - 1)
            # Exit on left wall
            elif exitDoorWall == 2:
                exitDoorPos = (topX, topY + self._rand_int(1, sizeY - 1))
            # Exit on north wall
            elif exitDoorWall == 3:
                exitDoorPos = (topX + self._rand_int(1, sizeX - 1), topY)
            else:
                assert False

            # Recursively create the other rooms
            success = self._placeRoom(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos,
            )

            if success:
                break

        return True

    def gen_obs(self):
        obs = super().gen_obs()
        obs.pop('mission')
        return obs

    def reset(self, **kwargs):
        total_size = self.width * self.height
        # Calculate the number of unique states visited by the agent
        unique_states_visited = np.count_nonzero(self.states)

        # Calculate the percentage of the environment the agent has visited
        self.percentage_visited = (unique_states_visited / total_size) * 100

        self.states = np.full((self.width, self.height), 0)
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        obs, _ = super().reset(**kwargs)
        self.intrinsic_reward = 0
        self.done = False
        return obs, {}


def custom_trial_name(trial):
    """
    Creates a custom trial name based on the configuration.

    Args:
        trial: The trial object from Ray Tune

    Returns:
        str: Custom trial name including LSTM size
    """
    env_config = trial.config.get("env_config", {})
    enable_dowham_reward_v1 = env_config.get("enable_dowham_reward_v1", False)
    enable_dowham_reward_v2 = env_config.get("enable_dowham_reward_v2", False)
    enable_count_based = env_config.get("enable_count_based", False)
    enable_rnd = env_config.get("enable_rnd", False)
    train_batch_size = trial.config.get("train_batch_size", "unknown")
    fc = trial.config.get("model", {}).get("fcnet_hiddens", "unknown")
    lstm = trial.config.get("model", {}).get("use_lstm", "unknown")
    grad_clip = trial.config.get("grad_clip", "unknown")
    vf_loss_coeff = trial.config.get("vf_loss_coeff", "unknown")
    entropy_coeff = trial.config.get("entropy_coeff", "unknown")

    if enable_dowham_reward_v1:
        return f"DoWhaMV1_batch{train_batch_size}{fc}{grad_clip}"
    if enable_dowham_reward_v2:
        return f"DoWhaMV2_batch{train_batch_size}Ent{entropy_coeff}VF{vf_loss_coeff}Lstm{lstm}{grad_clip}"
    elif enable_count_based:
        return f"CountBased_batch{train_batch_size}{fc}{grad_clip}"
    elif enable_rnd:
        return f"RND_batch{train_batch_size}{fc}{grad_clip}"
    else:
        return f"Default_batch{train_batch_size}Ent{entropy_coeff}VF{vf_loss_coeff}Lstm{lstm}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom training script")
    parser.add_argument('--num_rollout_workers', type=int, help='The number of rollout workers', default=1)
    parser.add_argument('--num_envs_per_worker', type=int, help='The number of environments per worker', default=1)
    parser.add_argument('--num_gpus', type=int, help='The number of GPUs to use', default=0)
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True, num_gpus=args.num_gpus, include_dashboard=False, log_to_driver=True,
             num_cpus=10, runtime_env={
            "env_vars": {
                "RAY_DISABLE_WORKER_STARTUP_LOGS": "0",
                "RAY_LOG_TO_STDERR": "0",
            }
        })

    env_type = CustomEnv.Environments.multi_room

    register_env("CustomPlaygroundCrossingEnv-v0",
                 lambda config:
                 RGBImgObsWrapper(CustomEnv(**config)))

    config = (
        PPOConfig()
        .training(
            use_critic=True,
            # use_gae=True,
            # entropy_coeff=0.02,
            # lambda_=0.95,
            # gamma=0.99,
            train_batch_size_per_learner=1024,
            lr=0.0004,
            # num_epochs=10,
            # lr=0.00025,
            # clip_param=0.2,
            grad_clip=5.0,
            vf_loss_coeff=0.01,
            # use_gae=True,
            # gamma=0.99,
            # lambda_=0.98,
            # kl_target=0.02,
            # kl_coeff=0.1,
            entropy_coeff=0.01,
            model={
                "fcnet_hiddens": [1024, 1024],
                "post_fcnet_hiddens": [1024, 1024],
                "dim": 88,
                "conv_filters": [
                    [16, [8, 8], 8],
                    [128, [9, 9], 1],
                ],
                "vf_share_layers": False,
            },
        )
        # .rl_module(
        #     model_config={
        #         "fcnet_hiddens": [256, 256],
        #         # "conv_filters": None,
        #         "conv_filters": [
        #             [16, [3, 3], 1],
        #             [32, [3, 3], 1],
        #             [64, [3, 3], 1],
        #         ],
        #         "vf_share_layers": False, },
        #     rl_module_spec=DefaultPPOTorchRLModule(
        #         observation_space=env.observation_space,
        #         action_space=env.action_space,
        #         model_config=DefaultModelConfig(fcnet_hiddens=[64, 64]),
        #         catalog_class=PPOCatalog(observation_space=env.observation_space, action_space=env.action_space,
        #                                  model_config_dict={
        #                                      "vf_share_layers": False,
        #                                      "head_fcnet_hiddens": [64, 64],
        #                                      "head_fcnet_activation": "relu",
        #                                  }),
        #     )
        # )
        .learners(
            num_learners=2,
            num_gpus_per_learner=args.num_gpus / 6,
        )
        .experimental(
            _disable_preprocessor_api=True, )
        .environment(
            env="CustomPlaygroundCrossingEnv-v0",
            disable_env_checking=True,
            env_config={
                "enable_dowham_reward_v2": True,
                "env_type": env_type
            },
        )
        .env_runners(
            num_env_runners=args.num_rollout_workers,
            num_envs_per_env_runner=args.num_envs_per_worker,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0
        )
        .framework("torch")
        .debugging(
            fake_sampler=False,
        ).api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        ).callbacks(CustomCallback)
        .evaluation(
            evaluation_interval=5,
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
            evaluation_parallel_to_training=False,
            evaluation_sample_timeout_s=120,
        )
    )

    algo = config.build_algo()

    checkpoint_config = CheckpointConfig(
        num_to_keep=5,
        checkpoint_frequency=5,
        checkpoint_at_end=True,
        checkpoint_score_attribute="evaluation/env_runners/episode_return_mean",
        checkpoint_score_order="max"
    )

    trails = [
        {
            **copy.deepcopy(config),
            "env_config": {
                "enable_dowham_reward_v2": True,
                "env_type": env_type
            },
            "fcnet_activation": "tanh",
            "post_fcnet_activation": "tanh",
            "lr": 0.0001,
            "grad_clip": 2.0,
        },
        {
            **copy.deepcopy(config),
            "env_config": {
                "enable_dowham_reward_v2": True,
                "env_type": env_type
            },
            "fcnet_activation": "tanh",
            "post_fcnet_activation": "tanh",
            "lr": 0.0001,
            "grad_clip": 2.0,
            "fcnet_hiddens": [512, 512],
            "post_fcnet_hiddens": [512, 512],
        },
        {
            **copy.deepcopy(config),
            "env_config": {
                "enable_dowham_reward_v2": True,
                "env_type": env_type
            },
            "fcnet_activation": "tanh",
            "post_fcnet_activation": "tanh",
            "lr": 0.0001,
            "grad_clip": 2.0,
            "fcnet_hiddens": [256, 256],
            "post_fcnet_hiddens": [256, 256],
        },
    ]

    trail = tune.run(
        "PPO",  # Specify the RLlib algorithm
        config=tune.grid_search(
            trails
        ),
        stop={
            "timesteps_total": 5_000_000,
        },
        checkpoint_config=checkpoint_config,
        trial_name_creator=custom_trial_name,  # Custom trial name
        verbose=2,  # Display detailed logs
        num_samples=1,  # Only one trial
        log_to_file=True,
        resume="AUTO",
        max_failures=5
    )
