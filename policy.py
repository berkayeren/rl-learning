from collections import deque

import gym
import minigrid
import numpy as np
from ray import tune, air
from ray.rllib.algorithms import ppo, Algorithm
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.numpy import one_hot
from ray.rllib.utils.test_utils import framework_iterator
from ray.tune.registry import register_env

tf1, tf, tfv = try_import_tf()


class CustomTFPolicy(TFPolicy):
    def __init__(self, observation_space, action_space, config):
        super(CustomTFPolicy, self).__init__(observation_space, action_space, config)

    def compute_actions(self, obs_batch, **kwargs):
        model_out, _ = self.model.forward({"obs": obs_batch}, [], None)
        return np.argmax(model_out.numpy(), axis=1), [], {}


class MyAlgo(Algorithm):
    def get_default_policy_class(self, config):
        return CustomTFPolicy


class OneHotWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, vector_index, framestack):
        super().__init__(env)
        self.framestack = framestack
        # 49=7x7 field of vision; 11=object types; 6=colors; 3=state types.
        # +4: Direction.
        self.single_frame_dim = 49 * (11 + 6 + 3) + 4
        self.init_x = None
        self.init_y = None
        self.x_positions = []
        self.y_positions = []
        self.x_y_delta_buffer = deque(maxlen=100)
        self.vector_index = vector_index
        self.frame_buffer = deque(maxlen=self.framestack)
        for _ in range(self.framestack):
            self.frame_buffer.append(np.zeros((self.single_frame_dim,)))

        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=(self.single_frame_dim * self.framestack,), dtype=np.float32
        )

    def observation(self, obs):
        # Debug output: max-x/y positions to watch exploration progress.
        if self.step_count == 0:
            for _ in range(self.framestack):
                self.frame_buffer.append(np.zeros((self.single_frame_dim,)))
            if self.vector_index == 0:
                if self.x_positions:
                    max_diff = max(
                        np.sqrt(
                            (np.array(self.x_positions) - self.init_x) ** 2
                            + (np.array(self.y_positions) - self.init_y) ** 2
                        )
                    )
                    self.x_y_delta_buffer.append(max_diff)
                    print(
                        "100-average dist travelled={}".format(
                            np.mean(self.x_y_delta_buffer)
                        )
                    )
                    self.x_positions = []
                    self.y_positions = []
                self.init_x = self.agent_pos[0]
                self.init_y = self.agent_pos[1]

        # Are we carrying the key?
        # if self.carrying is not None:
        #    print("Carrying KEY!!")

        self.x_positions.append(self.agent_pos[0])
        self.y_positions.append(self.agent_pos[1])

        # One-hot the last dim into 11, 6, 3 one-hot vectors, then flatten.
        objects = one_hot(obs[:, :, 0], depth=11)
        colors = one_hot(obs[:, :, 1], depth=6)
        states = one_hot(obs[:, :, 2], depth=3)
        # Is the door we see open?
        # for x in range(7):
        #    for y in range(7):
        #        if objects[x, y, 4] == 1.0 and states[x, y, 0] == 1.0:
        #            print("Door OPEN!!")

        all_ = np.concatenate([objects, colors, states], -1)
        all_flat = np.reshape(all_, (-1,))
        direction = one_hot(np.array(self.agent_dir), depth=4).astype(np.float32)
        single_frame = np.concatenate([all_flat, direction])
        self.frame_buffer.append(single_frame)
        return obs.get('image', {})


if __name__ == "__main__":
    tf1, tf, tfv = try_import_tf()


    def env_maker(config):
        name = config.get("name", "MiniGrid-Empty-5x5-v0")
        framestack = config.get("framestack", 4)
        env = gym.make(name)
        # Make it impossible to reach goal by chance.
        env = gym.wrappers.TimeLimit(env, max_episode_steps=15)
        # Only use image portion of observation (discard goal and direction).
        env = minigrid.wrappers.FlatObsWrapper(env)
        # env = OneHotWrapper(
        #     env,
        #     config.vector_index if hasattr(config, "vector_index") else 0,
        #     framestack=framestack,
        # )
        return env


    register_env("mini-grid", env_maker)
    CONV_FILTERS = [[16, [11, 11], 3], [32, [9, 9], 3], [64, [5, 5], 3]]
    ModelCatalog.register_custom_model("my_model", CustomTFPolicy)
    config = (
        ppo.PPOConfig()
        .environment(
            "mini-grid",
            env_config={
                # Also works with:
                # - MiniGrid-MultiRoom-N4-S5-v0
                # - MiniGrid-MultiRoom-N2-S4-v0
                "name": "MiniGrid-Empty-8x8-v0",
                "framestack": 1,  # seems to work even w/o framestacking
            },
        )
        .rollouts(num_envs_per_worker=4, num_rollout_workers=0)
        .training(
            model={
                "dim": 42,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                'conv_filters': CONV_FILTERS,
            },
            num_sgd_iter=8,
        )
        .exploration(
            exploration_config={
                "type": "Curiosity",
                # For the feature NN, use a non-LSTM fcnet (same as the one
                # in the policy model).
                "eta": 0.1,
                "lr": 0.0003,  # 0.0003 or 0.0005 seem to work fine as well.
                "feature_dim": 64,
                # No actual feature net: map directly from observations to feature
                # vector (linearly).
                "feature_net_config": {
                    "fcnet_hiddens": [],
                    "fcnet_activation": "relu",
                },
                "sub_exploration": {
                    "type": "StochasticSampling",
                },
            }
        )
    )

    min_reward = 0.001
    stop = {
        "training_iteration": 25,
        "episode_reward_mean": min_reward,
    }
    for _ in framework_iterator(config, frameworks="torch"):
        # To replay:
        # algo = ppo.PPO(config=config)
        # algo.restore("[checkpoint file]")
        # env = env_maker(config["env_config"])
        # obs, info = env.reset()
        # for _ in range(10000):
        #     obs, reward, done, truncated, info = env.step(
        #         algo.compute_single_action(s)
        #     )
        #     if done:
        #         obs, info = env.reset()
        #     env.render()

        results = tune.Tuner(
            "PPO",
            param_space=config,
            run_config=air.RunConfig(stop=stop, verbose=1),
        ).fit()
        # check_learning_achieved(results, min_reward)
        # iters = results.get_best_result().metrics["training_iteration"]
        # print("Reached in {} iterations.".format(iters))
