from __future__ import annotations

from enum import IntEnum
from typing import Union, Optional, Dict

import gymnasium as gym
import numpy as np
import ray
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Lava, Wall
from minigrid.envs import EmptyEnv
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


class CustomEnv(EmptyEnv):
    class Environments(IntEnum):
        empty = 0
        crossing = 1

    class NavigationOnlyActions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        done = 6

    def __init__(self, **kwargs):
        self.action = None
        self.reward_range = (0, 1)
        self.max_steps = 200
        self.env_type = kwargs.pop("env_type", CustomEnv.Environments.crossing)
        self.tile_size = 24
        self.highlight = False

        super().__init__(size=9, tile_size=self.tile_size, highlight=self.highlight, **kwargs)

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
        self.action = action
        obs, reward, terminated, truncated, _ = super().step(action)
        self.intrinsic_reward = reward
        self.done = terminated
        return obs, reward, terminated, truncated, {}

    def _gen_grid(self, width, height):
        if self.env_type == CustomEnv.Environments.crossing:
            self.crossing_env(width, height)
        elif self.env_type == CustomEnv.Environments.empty:
            self.empty_env_random_goal(width, height)

    def img_gen_obs(self):
        obs = super().gen_obs()
        obs.pop('mission')
        return obs

    def img_observation(self, obs):
        rgb_img = self.get_frame(
            highlight=self.highlight, tile_size=self.tile_size
        )

        # from PIL import Image
        # img = Image.fromarray(rgb_img)
        # img.save("debug_image.png")

        return {'direction': obs.get('direction'), "image": rgb_img.flatten()}

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

    def gen_obs(self):
        obs = super().gen_obs()
        obs.pop('mission')
        return {
            **obs}

    def reset(self, **kwargs):
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        obs, _ = super().reset(**kwargs)
        self.intrinsic_reward = 0
        self.done = False
        return obs, {}


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, num_gpus=0, include_dashboard=False, log_to_driver=True,
             num_cpus=10, runtime_env={
            "env_vars": {
                "RAY_DISABLE_WORKER_STARTUP_LOGS": "0",
                "RAY_LOG_TO_STDERR": "0",
            }
        })

    register_env("CustomPlaygroundCrossingEnv-v0",
                 lambda config:
                 RGBImgObsWrapper(CustomEnv(**config)))

    env = RGBImgObsWrapper(CustomEnv(env_type=CustomEnv.Environments.crossing))
    obs = env.reset()

    config = (
        PPOConfig()
        .training(
            use_critic=True,
            # use_gae=True,
            # entropy_coeff=0.02,
            # lambda_=0.95,
            # gamma=0.99,
            train_batch_size_per_learner=2000,
            lr=0.0004,
            # num_epochs=10,
            # lr=0.00025,
            # clip_param=0.2,
            grad_clip=5.0,
            vf_loss_coeff=0.01,
            entropy_coeff=0.01,
            model={
                "fcnet_hiddens": [256, 256],
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
            num_gpus_per_learner=0,
        ).resources(
            num_gpus=0,
        )
        .experimental(
            _disable_preprocessor_api=True,
        )
        .environment(
            env="CustomPlaygroundCrossingEnv-v0",
            observation_space=env.observation_space,
            action_space=env.action_space,
            disable_env_checking=True
        )
        .env_runners(num_env_runners=4, num_envs_per_env_runner=1, num_gpus_per_env_runner=0)
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

    trail = tune.run(
        "PPO",  # Specify the RLlib algorithm
        config=config,
        stop={
            "timesteps_total": 10_000_000,
        },
        checkpoint_config=checkpoint_config,
        verbose=2,  # Display detailed logs
        num_samples=1,  # Only one trial
        log_to_file=True,
        resume="AUTO",
        max_failures=5
    )
