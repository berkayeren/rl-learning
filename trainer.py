from __future__ import annotations

import argparse
import collections
import copy
import hashlib
from enum import IntEnum
from typing import Union, Optional, Dict

import gymnasium as gym
import numpy as np
import ray
import torch
from gymnasium import spaces
from matplotlib import pyplot as plt
from minigrid.core.constants import COLOR_NAMES

VIEW_SIZE = 7
# Pivot at bottom-center of the egocentric view (row 6, col 3)
PIVOT_ROW = VIEW_SIZE - 1
PIVOT_COL = VIEW_SIZE // 2
BASE_OFFSETS = np.array([
    [(PIVOT_ROW - i, j - PIVOT_COL) for j in range(VIEW_SIZE)]
    for i in range(VIEW_SIZE)
])
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Lava, Wall, Door
from minigrid.envs import MultiRoom
from minigrid.wrappers import RGBImgObsWrapper, FullyObsWrapper
from ray import tune, train
from ray.air import FailureConfig
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module import RLModule
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType, PolicyID
from ray.tune import register_env, CheckpointConfig
from ray.tune.search import BasicVariantGenerator

from environments.minigrid_wrapper import PositionBasedWrapper
from environments.empty import EmptyEnv
from intrinsic_motivation.count_based import CountExploration
from intrinsic_motivation.dowham_v2 import DoWhaMIntrinsicRewardV2
from intrinsic_motivation.dowham_v1 import DoWhaMIntrinsicRewardV1
from intrinsic_motivation.rnd import RNDModule

import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX


class CustomCallback(RLlibCallback):
    counter = 1

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
        episode.custom_metrics["termination_reward"] = env.termination_reward

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
        # episode.custom_metrics["percentage_visited"] = env.percentage_visited
        # episode.custom_metrics["percentage_history"] = env.percentage_history.count(True)
        self.counter += 1

        if self.counter % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # try:
        #     if self.counter % 100 == 0:
        #         plot_heatmap(env, f"heatmaps/heat_map{env_index}{self.counter}.png")
        #         env.states = np.full((env.width, env.height), 0)
        # except Exception:
        #     pass

    def on_episode_end(
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
        episode.custom_metrics["percentage_history"] = env.percentage_history.count(True)


def plot_heatmap(env, filename="visit_heatmap.png"):
    """
    Plots a heatmap of agent visits, overlaying walls and the goal position.
    """
    heatmap_data = np.flipud(env.states.T)  # Flip for correct orientation

    plt.figure(figsize=(6, 6))
    plt.title("Agent Visit Heatmap")

    # Plot heatmap with visit counts
    plt.imshow(heatmap_data, cmap="hot", origin="lower", alpha=0.5)  # Heatmap semi-transparent

    for x in range(env.width):
        for y in range(env.height):
            count = env.states[x, y]
            if count > 0:  # Only show counts for visited cells
                plt.text(x, env.height - y - 1, str(count), ha='center', va='center', color='white', fontsize=8)

    # Overlay walls in black
    for x in range(env.width):
        for y in range(env.height):
            if isinstance(env.grid.get(x, y), Wall):  # Check if cell is a wall
                plt.scatter(x, env.height - y - 1, color='black', s=40, marker='s')
            if isinstance(env.grid.get(x, y), Door):  # Check if cell is a wall
                plt.scatter(x, env.height - y - 1, color="yellow", s=80, marker='D', edgecolors="black",
                            linewidth=1.2)

    # Overlay goal position in green
    if hasattr(env, "goal_pos") and env.goal_pos:
        goal_x, goal_y = env.goal_pos
        plt.scatter(goal_x, env.height - goal_y - 1, color='lime', s=100, marker='*', edgecolors="black", linewidth=1.5)

    # Add colorbar
    plt.colorbar(label="Visit Count")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(False)

    # Save the image
    plt.savefig(filename)
    plt.close()


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
        self.termination_reward = 0
        self.env_type = kwargs.pop("env_type", CustomEnv.Environments.empty)
        self.enable_dowham_reward_v1 = kwargs.pop('enable_dowham_reward_v1', False)
        self.enable_dowham_reward_v2 = kwargs.pop('enable_dowham_reward_v2', False)
        self.enable_count_based = kwargs.pop('enable_count_based', False)
        self.enable_rnd = kwargs.pop('enable_rnd', False)
        self.direction_obs = kwargs.pop('direction_obs', True)
        self.max_steps = kwargs.pop('max_steps', 200)
        self.conv_filter = kwargs.pop('conv_filter', False)
        self.is_partial_obs = kwargs.pop('is_partial_obs', True)
        self.highlight = kwargs.pop('highlight', False)
        print(f"Enable Dowham Reward V2: {self.enable_dowham_reward_v2}")

        self.percentage_visited = 0.0
        self.percentage_history = collections.deque(maxlen=100)
        self.action = None
        self.reward_range = (-1, 1)
        self.dowham_reward = None
        self.tile_size = 24
        self.see_through_walls = False
        super().__init__(
            size=19,
            tile_size=self.tile_size,
            highlight=self.highlight,
            max_steps=self.max_steps,
            see_through_walls=False,
            # render_mode="human",
            **kwargs)

        self.states = np.full((self.width, self.height), 0)

        # Later in the method, after other initializations
        if self.enable_rnd:
            print("RND Exploration Enabled")
            self.rnd = RNDModule(input_dim=148, embed_dim=32,  # Match fcnet_hiddens size
                                 hidden_size=32, reward_scale=1.0)
        if self.enable_dowham_reward_v1:
            print("Enable Dowham Reward V1")
            self.reward_range = (0, 1)
            self.dowham_reward = DoWhaMIntrinsicRewardV1(eta=40, H=1, tau=0.5)
        if self.enable_dowham_reward_v2:
            print("Enable Dowham Reward V2")
            self.reward_range = (-1, 1)
            self.dowham_reward = DoWhaMIntrinsicRewardV2(eta=40, H=1, tau=0.5)
        if self.enable_count_based:
            print(f"Count Based Exploration Enabled")
            self.count_based = CountExploration(self, gamma=0.99, epsilon=0.1, alpha=0.1)

        # new_image_space = spaces.Box(
        #     low=0,
        #     high=255,
        #     shape=(
        #         self.height * self.tile_size,
        #         self.width * self.tile_size,
        #         3,
        #     ),
        #     dtype="uint8",
        # )
        #
        # self.observation_space = spaces.Dict(
        #     {'direction': self.observation_space.spaces['direction'], "image": new_image_space}
        # )

        print(f"Environment Type: {CustomEnv.Environments(self.env_type).name}")

        if self.env_type == CustomEnv.Environments.crossing:
            self.obstacle_type = Wall
            self.num_crossings = 1
            self.max_door = 1

        if self.env_type == CustomEnv.Environments.empty:
            self.max_door = 0

        if self.env_type == CustomEnv.Environments.four_rooms:
            self.max_door = 4

        if self.env_type == CustomEnv.Environments.multi_room:
            self.max_door = 3

        self.intrinsic_reward = 0
        self.done = False

        if self.max_door > 0:
            # For environments with doors (crossing, four_rooms, multi_room)
            self.observation_space = spaces.Dict({
                'agent_pos': spaces.Box(low=0, high=max(self.width, self.height),
                                        shape=(2,), dtype=np.int32),
                'agent_dir': spaces.Discrete(4),
                'goal_pos': spaces.Box(low=0, high=max(self.width, self.height),
                                       shape=(2,), dtype=np.int32),
                'door_pos': spaces.Box(low=0, high=max(self.width, self.height),
                                       shape=(self.max_door, 2), dtype=np.int32),
                'door_state': spaces.Box(low=0, high=1,
                                         shape=(self.max_door,), dtype=np.int32)
            })
        else:
            # For environments without doors (empty)
            self.observation_space = spaces.Dict({
                'agent_pos': spaces.Box(low=0, high=max(self.width, self.height),
                                        shape=(2,), dtype=np.int32),
                'agent_dir': spaces.Discrete(4),
                'goal_pos': spaces.Box(low=0, high=max(self.width, self.height),
                                       shape=(2,), dtype=np.int32)
            })

    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """
        # Ensures minimum reward of 0.5 even at max steps
        # Provides better gradient between optimal and worst-case performance
        return 1.0 - 0.9 * (self.step_count / self.max_steps)

    def transform_coords(self, x, y, agent_dir):
        if agent_dir == 0:  # Right (→), no change
            return x, y
        elif agent_dir == 1:  # Down (↓), rotate clockwise 90°
            return -y, x
        elif agent_dir == 2:  # Left (←), rotate 180°
            return -x, -y
        elif agent_dir == 3:  # Up (↑), rotate counter-clockwise 90°
            return y, -x
        return None

    def extract_visible_coords_from_obs(self, obs, agent_pos, agent_dir):
        """
        Map the agent's 7×7 egocentric type-mask to global coordinates.
        Includes all non-unseen, non-wall cells. Uses BASE_OFFSETS (East-facing)
        and transform_coords to rotate for any agent_dir.
        """

        # 1) Extract the raw type mask (shape VIEW_SIZE×VIEW_SIZE) and filter
        raw_img = obs['image'] if isinstance(obs, dict) else obs
        agent_col = 7 // 2
        agent_row = 7 - 1
        raw_img[agent_col, agent_row, 0] = OBJECT_TO_IDX['agent']
        mask = raw_img[:, :, 0].T  # Transpose to (height, width)
        valid = (mask != OBJECT_TO_IDX['unseen']) & (mask != OBJECT_TO_IDX['wall']) & (mask != OBJECT_TO_IDX['agent'])
        coords = np.argwhere(valid)

        # 2) Map each local (i,j) to global coords
        ax, ay = agent_pos
        visible = []
        for i, j in coords:
            # East-facing offset (dx,dy) from BASE_OFFSETS
            dx, dy = BASE_OFFSETS[i, j]
            # Rotate into actual direction
            rdx, rdy = self.transform_coords(dx, dy, agent_dir)
            wx, wy = ax + rdx, ay + rdy

            # Boundary check
            if 0 <= wx < self.width and 0 <= wy < self.height:
                visible.append((int(wx), int(wy)))

        return visible

    def step(self, action: int):
        self.states[self.agent_pos[0]][self.agent_pos[1]] += 1
        self.action = action
        current_obs_hash, current_obs = self.img_observation()
        prev_pos = (self.agent_pos[0], self.agent_pos[1]) if isinstance(self.agent_pos, np.ndarray) else self.agent_pos
        prev_dir = self.agent_dir
        obs, reward, terminated, truncated, _ = super().step(action)
        next_obs_hash, next_obs = self.img_observation()
        next_pos = (self.agent_pos[0], self.agent_pos[1]) if isinstance(self.agent_pos, np.ndarray) else self.agent_pos
        next_dir = self.agent_dir
        # print(f"Agent Pos: {self.agent_pos[0]}, {self.agent_pos[1]}")
        if self.enable_dowham_reward_v1 or self.enable_dowham_reward_v2:
            curr_view = self.extract_visible_coords_from_obs(current_obs, prev_pos, prev_dir)
            next_view = self.extract_visible_coords_from_obs(next_obs, next_pos, next_dir)

            # print(f"Agent Pos: {prev_pos}, Agent Dir: {prev_dir}, Current View: {curr_view}")
            # print(f"Agent Pos: {self.agent_pos}, Agent Dir: {self.agent_dir}, Next View: {next_view}")
            self.dowham_reward.update_state_visits(current_obs_hash, next_obs_hash)
            state_changed = current_obs_hash != next_obs_hash or prev_dir != next_dir
            self.dowham_reward.update_usage(current_obs_hash, action)

            self.dowham_reward.update_effectiveness(
                current_obs_hash,
                action,
                next_obs_hash,
                state_changed
            )

            if self.enable_dowham_reward_v2:
                self.intrinsic_reward = self.dowham_reward.calculate_intrinsic_reward(
                    current_obs_hash,
                    action,
                    next_obs_hash,
                    state_changed,
                    curr_view,
                    next_view,
                    next_pos
                )
            else:
                self.intrinsic_reward = self.dowham_reward.calculate_intrinsic_reward(
                    current_obs_hash,
                    action,
                    next_obs_hash,
                    state_changed
                )

        if self.enable_count_based:
            self.intrinsic_reward = self.count_based.update((prev_pos[0], prev_pos[1]), action, reward, self.goal_pos)
        if self.enable_rnd:
            self.rnd_reward(obs)

        reward = reward * 0.05
        if terminated:
            reward += self._reward()
            self.termination_reward = reward
        else:
            reward += self.intrinsic_reward * 0.05

        self.done = terminated
        return obs, reward, terminated, truncated, {}

    def rnd_reward(self, obs):
        # First, convert observation to a format suitable for RND
        if isinstance(obs, dict):
            # Flatten dict observation
            obs_list = []
            for key, value in obs.items():
                if isinstance(value, (int, np.int32, np.int64)):
                    value = np.array([value])
                elif not isinstance(value, np.ndarray):
                    value = np.array(value)
                obs_list.append(value.flatten())
            flat_obs = np.concatenate(obs_list)
        else:
            flat_obs = np.array(obs).flatten()

        flat_obs = flat_obs.astype(np.float32)
        self.rnd.update_obs_normalizer(flat_obs)
        # Calculate intrinsic reward
        self.intrinsic_reward = self.rnd.compute_intrinsic_reward(flat_obs)
        # Update predictor network
        self.rnd.update_predictor(flat_obs)

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
        if not self.is_partial_obs:
            if self.direction_obs:
                rgb_img = self.get_frame(
                    highlight=self.highlight, tile_size=self.tile_size
                )
            else:
                agent_dir = self.agent_dir
                self.agent_dir = 0
                rgb_img = self.get_frame(
                    highlight=self.highlight, tile_size=self.tile_size
                )
                self.agent_dir = agent_dir
        else:
            rgb_img = self.gen_obs()["image"]

        sample_hash = hashlib.sha256()

        # from PIL import Image
        # img = Image.fromarray(rgb_img)
        # img.save("debug_image.png")

        # Convert the full grid to a list for hashing
        full_grid_list = rgb_img.flatten().tolist()

        to_encode = [full_grid_list, self.agent_pos, self.agent_dir]

        # Update the hash with each element
        for item in to_encode:
            sample_hash.update(str(item).encode("utf8"))

        # Return the hashed value
        return sample_hash.hexdigest()[:size], rgb_img

    def crossing_env(self, width, height):
        import itertools as itt
        assert width % 2 == 1 and height % 2 == 1  # odd size
        self.obstacle_type = Wall
        self.num_crossings = 1

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)
        self.goal_pos = (width - 2, height - 2)

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
            self.grid.set(i, j, Door(color="yellow", is_open=False))
        self.max_door = 1
        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

    def empty_env_random_goal(self, width, height):
        self.max_door = 0
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        # Get grid size from environment
        grid_size = self.width  # Assuming width == height
        # self.put_obj(Goal(), 17, 17)
        # self.goal_pos = (17, 17)
        # Randomly assign a new goal position (excluding (1,1))
        while True:
            self.goal_pos = (np.random.randint(1, grid_size - 2), np.random.randint(1, grid_size - 2))
            if self.goal_pos != (1, 1):  # Ensure it's not the starting position
                self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])
                break

    def four_rooms(self, width, height):
        self.max_door = 4
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
                    self.grid.set(pos[0], pos[1], Door(color="yellow", is_open=False))

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(pos[0], pos[1], Door(color="yellow", is_open=False))

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
        """
        Example of a 19×19 fixed, maze-like layout with several 'yellow' doors
        and a goal in the lower corridor.
        """
        # Force the grid size to 19×19.
        width = 19
        height = 19
        self.grid = Grid(width, height)

        # 1) Surrounding outer walls
        self.grid.wall_rect(1, 0, 5, 7)
        self.grid.wall_rect(0, 6, 7, 5)
        self.grid.wall_rect(6, 8, 9, 5)
        self.grid.wall_rect(14, 8, 5, 11)
        self.grid.set(3, 6, Door(color="yellow", is_open=False))
        self.grid.set(6, 9, Door(color="yellow", is_open=False))
        self.grid.set(14, 10, Door(color="yellow", is_open=False))
        self.max_door = 3
        # ----------------------------------------------------------------------
        # 4) Agent start in the top-left corridor
        # ----------------------------------------------------------------------
        self.agent_pos = (2, 1)
        self.agent_dir = 0  # facing right

        # ----------------------------------------------------------------------
        # 5) Goal in the lower corridor (the green square)
        # ----------------------------------------------------------------------
        self.put_obj(Goal(), 17, 17)
        self.goal_pos = (17, 17)

    def _multi_room(self, width, height):
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
        #
        # [x,y, direction] [goal_X, goal_Y, goal,]
        # degisebilecek tüm bilgileri agent a verecek bir wrapper yaz.
        obs = super().gen_obs()

        if self.conv_filter or self.enable_rnd:
            obs.pop('mission')

        return obs

    def reset(self, **kwargs):
        total_size = self.width * self.height
        # Calculate the number of unique states visited by the agent
        unique_states_visited = np.count_nonzero(self.states)

        # Calculate the percentage of the environment the agent has visited
        self.percentage_visited = (unique_states_visited / total_size) * 100
        self.percentage_history.append(self.done)
        self.states = np.full((self.width, self.height), 0)

        if self.enable_dowham_reward_v2 or self.enable_dowham_reward_v1:
            self.dowham_reward.reset_episode()

        # if len(self.percentage_history) == 100 and self.percentage_history.count(
        #         True) >= 90 and self.env_type != CustomEnv.Environments.multi_room:
        #     self.env_type = (self.env_type + 1) % len(CustomEnv.Environments)
        #     print(f"Environment Type Changed to: {CustomEnv.Environments(self.env_type).name}")

        if self.env_type == CustomEnv.Environments.multi_room:
            self.agent_pos = (2, 1)
        else:
            self.agent_pos = (1, 1)

        self.agent_dir = 0
        obs, _ = super().reset(**kwargs)
        self.intrinsic_reward = 0
        self.termination_reward = 0
        self.done = False
        return obs, {}


def custom_trial_name(trial):
    """
    Creates a custom trial name based on the configuration.

    Args:
        trial: The trial object from Ray Tune

    Returns:
        str: Custom trial name including exploration type and environment type
    """
    env_config = trial.config.get("env_config", {})
    enable_dowham_reward_v1 = env_config.get("enable_dowham_reward_v1", False)
    enable_dowham_reward_v2 = env_config.get("enable_dowham_reward_v2", False)
    enable_count_based = env_config.get("enable_count_based", False)
    enable_rnd = env_config.get("enable_rnd", False)
    env_type = env_config.get("env_type", "unknown")
    train_batch_size = trial.config.get("train_batch_size", "unknown")
    fc = trial.config.get("model", {}).get("fcnet_hiddens", "unknown")
    grad_clip = trial.config.get("grad_clip", "unknown")

    exploration_type = "Default"
    if enable_dowham_reward_v1:
        exploration_type = "DoWhaMV1"
    elif enable_dowham_reward_v2:
        exploration_type = "DoWhaMV2"
    elif enable_count_based:
        exploration_type = "CountBased"
    elif enable_rnd:
        exploration_type = "RND"

    return f"{exploration_type}_{CustomEnv.Environments(env_type).name}_batch{train_batch_size}{fc}{grad_clip}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom training script")
    parser.add_argument('--num_rollout_workers', type=int, help='The number of rollout workers', default=1)
    parser.add_argument('--num_envs_per_worker', type=int, help='The number of environments per worker', default=1)
    parser.add_argument('--num_gpus', type=int, help='The number of GPUs to use', default=0)
    parser.add_argument('--num_samples', type=int, help='Number of samples', default=1)
    parser.add_argument('--timesteps_total', type=int, help='Timesteps Total', default=100_000_000)
    parser.add_argument('--max_steps', type=int, help='Max Time Steps', default=200)
    parser.add_argument('--conv_filter', action="store_true", help='Use convolutional layer or flat observation')
    parser.add_argument('--environment', type=str, help='Environment to choose', choices=[
        "empty",
        "crossing",
        "four_rooms",
        "multi_room",
    ], default="empty")
    parser.add_argument('--run_mode', type=str, choices=['experiment', 'hyperparameter_search'], required=True,
                        help='Specify whether to run an experiment or hyperparameter search')
    parser.add_argument('--trail_name', type=str, help='Custom trail name', default=None)

    args = parser.parse_args()

    ray.init(ignore_reinit_error=True, num_gpus=args.num_gpus, include_dashboard=False, log_to_driver=True,
             num_cpus=10, runtime_env={
            "env_vars": {
                "RAY_DISABLE_WORKER_STARTUP_LOGS": "0",
                "RAY_LOG_TO_STDERR": "0",
            }
        })

    env_type = CustomEnv.Environments[args.environment]

    if args.conv_filter:
        register_env("CustomPlaygroundCrossingEnv-v0",
                     lambda config:
                     RGBImgObsWrapper(FullyObsWrapper(CustomEnv(**config))))

        env = RGBImgObsWrapper(FullyObsWrapper(CustomEnv(env_type=env_type)))
        obs = env.reset()
    else:
        register_env("CustomPlaygroundCrossingEnv-v0",
                     lambda config:
                     PositionBasedWrapper(CustomEnv(**config)))

        env = PositionBasedWrapper(CustomEnv(env_type=env_type))
        obs = env.reset()

        # config = (
    #     PPOConfig()
    #     .training(
    #         gamma=0.99,  # Discount factor
    #         lr=0.0008679592813302736,  # Learning rate
    #         grad_clip=4.488759919509276,  # Gradient clipping
    #         grad_clip_by="global_norm",
    #         train_batch_size=128,  # Training batch size
    #         num_epochs=30,  # Number of training epochs
    #         minibatch_size=128,  # Mini-batch size for SGD
    #         shuffle_batch_per_epoch=True,
    #         use_critic=True,
    #         use_gae=True,  # Generalized Advantage Estimation
    #         use_kl_loss=True,
    #         kl_coeff=0.16108743826129673,  # KL divergence coefficient
    #         kl_target=0.01,  # Target KL divergence
    #         vf_loss_coeff=0.02633906005324078,  # Value function loss coefficient
    #         entropy_coeff=0.1,  # Entropy coefficient for exploration
    #         clip_param=0.25759466534505526,  # PPO clipping parameter
    #         vf_clip_param=10.0,  # Clipping for value function updates
    #         optimizer={
    #             "type": "RMSProp",
    #         },
    #         model={
    #             "fcnet_hiddens": [1024, 1024],
    #             "fcnet_activation": "tanh",
    #             "post_fcnet_hiddens": [512, 512],
    #             "post_fcnet_activation": "tanh",
    #             "conv_filters": [
    #                 [32, [8, 8], 8],  # 32 filters, 8x8 kernel, stride 8
    #                 [128, [11, 11], 1],  # 128 filters, 11x11 kernel, stride 1
    #             ],
    #             # "conv_filters": [
    #             #     [16, [8, 8], 4],
    #             #     [32, [4, 4], 2],
    #             #     [256, [11, 11], 1],
    #             # ],
    #             "conv_activation": "relu",
    #             "vf_share_layers": False,
    #             "framestack": True,
    #             "dim": 84,  # Resized observation dimension
    #             "grayscale": False,
    #             "zero_mean": True,
    #         }
    #     ).learners(
    #         num_learners=2,
    #         num_gpus_per_learner=args.num_gpus / 6,
    #     )
    #     .experimental(
    #         _disable_preprocessor_api=True, )
    #     .environment(
    #         env="CustomPlaygroundCrossingEnv-v0",
    #         env_config={
    #             "enable_dowham_reward_v2": True,
    #             "env_type": env_type
    #         },
    #         disable_env_checking=True,
    #         normalize_actions=True,
    #         clip_actions=False,
    #     )
    #     .env_runners(
    #         num_env_runners=args.num_rollout_workers,
    #         num_envs_per_env_runner=args.num_envs_per_worker,
    #         num_cpus_per_env_runner=0.5,
    #         num_gpus_per_env_runner=args.num_gpus / 6,
    #         batch_mode="complete_episodes",
    #     )
    #     .framework("torch")
    #     .debugging(
    #         fake_sampler=False,
    #     ).api_stack(
    #         enable_rl_module_and_learner=False,
    #         enable_env_runner_and_connector_v2=False,
    #     ).callbacks(CustomCallback)
    #     .evaluation(
    #         evaluation_interval=20,
    #         evaluation_duration=10,
    #         evaluation_duration_unit="episodes",
    #         evaluation_parallel_to_training=False,
    #         evaluation_sample_timeout_s=120,
    #     )
    # )
    config = (
        PPOConfig()
        .training(
            use_critic=True,
            use_kl_loss=False,
            # vtrace=True,
            gamma=0.99,  # Discount factor
            lr=1e-4,  # Learning rate
            train_batch_size_per_learner=1024,
            train_batch_size=1024,  # Larger batches
            # num_sgd_iter=2,  # More SGD iterations
            grad_clip=5.0,  # Tighter gradient clipping
            # optimizer={
            #     "type": "rmsprop",
            #     "momentum": 0.0,
            #     "epsilon": 0.01,
            # },
            # opt_type="rmsprop",
            # epsilon=0.01,
            # momentum=0.0,
            vf_loss_coeff=0.5,  # Value function loss coefficient
            entropy_coeff=0.005,
            model={
                "fcnet_hiddens": [32, 32],
                "post_fcnet_hiddens": [32],
                "fcnet_activation": "relu",
                # "conv_filters": [
                #     [32, [3, 3], 2],
                #     [64, [3, 3], 2],
                #     [64, [3, 3], 2],
                # ],
                "conv_filters": [
                    [32, [3, 3], 2],  # First Conv Layer: 32 filters, 3x3 kernel, stride 2
                    [32, [3, 3], 2],  # Second Conv Layer: 32 filters, 3x3 kernel, stride 2
                    [32, [3, 3], 2],  # Third Conv Layer: 32 filters, 3x3 kernel, stride 2
                    [32, [3, 3], 2],  # Third Conv Layer: 32 filters, 3x3 kernel, stride 2
                    [32, [3, 3], 2],  # Third Conv Layer: 32 filters, 3x3 kernel, stride 2
                ] if args.conv_filter else None,
                "conv_activation": "elu",
                "vf_share_layers": False,
                "framestack": False,
                "grayscale": False,
                "custom_preprocessor": None,
                "zero_mean": False,  # Normalize inputs with dim
                "use_lstm": False,  # Add recurrent layer for temporal dependencies
                "lstm_cell_size": 128,
                "max_seq_len": 33,
            }
        ).learners(
            num_gpus_per_learner=0.8 if args.num_gpus > 0 else 0,
            num_learners=1,
            num_cpus_per_learner=1,
        )
        .experimental(
            _disable_preprocessor_api=True, )
        .environment(
            env="CustomPlaygroundCrossingEnv-v0",
            env_config={
                "enable_dowham_reward_v2": False,
                "env_type": env_type
            },
            disable_env_checking=True,
            is_atari=False,
            # observation_space=env.observation_space,
            # action_space=env.action_space,
        )
        .env_runners(
            num_env_runners=args.num_rollout_workers,
            num_envs_per_env_runner=args.num_envs_per_worker,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
            # rollout_fragment_length=256,
            batch_mode="complete_episodes",  # Better for IMPALA
            # explore=True,
            # exploration_config={
            #     "type": "EpsilonGreedy",
            #     "initial_epsilon": 1.0,
            #     "final_epsilon": 0.01,
            #     "epsilon_timesteps": 1_000_000,
            # }
        )
        .framework("torch").resources(
            num_gpus=args.num_gpus,
            placement_strategy="SPREAD"
        )
        .debugging(
            fake_sampler=False,
        ).api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        ).callbacks(CustomCallback)
        .evaluation(
            evaluation_interval=200,
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
            evaluation_parallel_to_training=False,
            evaluation_sample_timeout_s=120,
        ).fault_tolerance(
            restart_failed_env_runners=True
        )
    )

    algo = config.build_algo()

    checkpoint_config = CheckpointConfig(
        num_to_keep=5,
        checkpoint_frequency=5,
        checkpoint_at_end=True,
        checkpoint_score_attribute="env_runners/episode_len_mean",
        checkpoint_score_order="min"
    )

    if args.run_mode == 'experiment':
        search_space = {
            **copy.deepcopy(config),
            "env_config": tune.grid_search([
                # Default PPO
                {"env_type": 3, "max_steps": args.max_steps, "conv_filter": args.conv_filter,
                 "enable_dowham_reward_v1": False, "enable_dowham_reward_v2": False, "enable_count_based": False,
                 "enable_rnd": True},
                # {"env_type": 1, "max_steps": args.max_steps, "conv_filter": args.conv_filter,
                #  "enable_dowham_reward_v1": True, "enable_count_based": False, "enable_rnd": False},
                # {"env_type": 3, "max_steps": args.max_steps, "conv_filter": args.conv_filter,
                #  "enable_dowham_reward_v1": False,
                #  "enable_dowham_reward_v2": True, "enable_count_based": False, "enable_rnd": False},
                # {"env_type": 2, "max_steps": args.max_steps, "conv_filter": args.conv_filter,
                #  "enable_dowham_reward_v2": False, "enable_count_based": False, "enable_rnd": False},
                # {"env_type": 3, "max_steps": args.max_steps, "conv_filter": args.conv_filter,
                #  "enable_dowham_reward_v2": False, "enable_count_based": False, "enable_rnd": False},
                #
                # # PPO with Count reward
                # {"env_type": 1, "max_steps": args.max_steps, "conv_filter": args.conv_filter,
                #  "enable_dowham_reward_v2": False, "enable_count_based": True, "enable_rnd": False},
                # {"env_type": 2, "max_steps": args.max_steps, "conv_filter": args.conv_filter,
                #  "enable_dowham_reward_v2": False, "enable_count_based": True, "enable_rnd": False},
                # {"env_type": 3, "max_steps": args.max_steps, "conv_filter": args.conv_filter,
                #  "enable_dowham_reward_v2": False, "enable_count_based": True, "enable_rnd": False},
                #
                # # PPO with RND reward
                # {"env_type": 1, "max_steps": args.max_steps, "conv_filter": args.conv_filter,
                #  "enable_dowham_reward_v2": False, "enable_count_based": False, "enable_rnd": True},
                # {"env_type": 2, "max_steps": args.max_steps, "conv_filter": args.conv_filter,
                #  "enable_dowham_reward_v2": False, "enable_count_based": False, "enable_rnd": True},
                # {"env_type": 3, "max_steps": args.max_steps, "conv_filter": args.conv_filter,
                #  "enable_dowham_reward_v2": False, "enable_count_based": False, "enable_rnd": True}
            ]),
            "seed": tune.grid_search(list(range(args.num_samples))),
        }

        trail = tune.run(
            "PPO",  # Specify the RLlib algorithm
            config=search_space,
            metric="env_runners/episode_len_mean",
            mode="min",
            stop={
                "timesteps_total": args.timesteps_total,
            },
            checkpoint_config=checkpoint_config,
            trial_name_creator=custom_trial_name,  # Custom trial name
            verbose=2,  # Display detailed logs
            num_samples=1,
            log_to_file=True,
            resume=True,
            max_failures=-1,
            reuse_actors=False,
            name=args.trail_name if hasattr(args, 'trail_name') and args.trail_name else None
        )
    elif args.run_mode == 'hyperparameter_search':
        trail = tune.Tuner(
            "PPO",  # Specify the RLlib algorithm
            param_space={
                **copy.deepcopy(config),
                "env_config": {
                    "enable_dowham_reward_v2": False,
                    "env_type": env_type,
                    "max_steps": args.max_steps,
                },
                "num_sgd_iter": tune.grid_search([3, 5, 10]),
            },
            tune_config=tune.TuneConfig(
                metric="env_runners/episode_reward_mean",  # Optimize for return
                mode="max",  # Maximize reward
                num_samples=args.num_samples,
                reuse_actors=True,
                search_alg=BasicVariantGenerator(),
                # Use Bayesian optimization
            ),
            run_config=train.RunConfig(stop={"timesteps_total": args.timesteps_total},
                                       failure_config=FailureConfig(max_failures=-1)),
        )
        # trail = tune.Tuner.restore(
        #     path="/Users/berkayeren/ray_results/IMPALA_2025-03-05_21-39-47",
        #     trainable="IMPALA",
        #     param_space={
        #         **copy.deepcopy(config),
        #         "env_config": {
        #             "enable_dowham_reward_v2": False,
        #             "env_type": env_type,
        #             "max_steps": 200,
        #         },
        #         "vf_loss_coeff": tune.grid_search([0.1, 0.2, 0.5]),
        #     },
        # )

        results = trail.fit()
        best_trial = results.get_best_result(metric="env_runners/episode_len_mean", mode="min")
        print("Best Hyperparameters:", best_trial.config)
