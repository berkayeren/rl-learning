import faulthandler

import numpy as np
import ray
import torch
from minigrid.core.actions import Actions
from ray.rllib.algorithms import PPOConfig, PPO
from ray.tune import register_env

from environments.minigrid_wrapper import PositionBasedWrapper
from trainer import CustomEnv

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, configure_logging=False)
    faulthandler.disable()

    env = PositionBasedWrapper(
        CustomEnv(env_type=CustomEnv.Environments.four_rooms, enable_dowham_reward_v2=False, direction_obs=False,
                  render_mode="human"))

    # Register the custom environment
    register_env("CustomPlaygroundCrossingEnv-v0",
                 lambda config:
                 PositionBasedWrapper(
                     CustomEnv(**config)))

    algo = PPO.from_checkpoint(
        r"/Users/berkayeren/ray_results/PPO_2025-03-27_12-15-15/Default_batch_0_2025-03-27_12-15-15/checkpoint_000016",
        config=PPOConfig().env_runners(
            num_env_runners=1,
            num_envs_per_env_runner=1,
        ).environment(env="CustomPlaygroundCrossingEnv-v0"))
    # algo = None
    # manual_control = ManualControl(env, seed=42)
    # manual_control.start()
    for episode in range(0, 10):
        action = 0
        reward = 0
        terminated = False
        truncated = False
        obs, _ = env.reset()

        while not (terminated or truncated):
            # Convert observation to tensor or the appropriate format
            if isinstance(obs, dict):
                # If using the PositionBasedWrapper
                # Convert dict values to tensors if they're numpy arrays
                obs_processed = {
                    k: torch.tensor(v) if isinstance(v, np.ndarray) else v
                    for k, v in obs.items()
                }
            else:
                # If using a different format
                obs_processed = torch.tensor(obs) if isinstance(obs, np.ndarray) else obs

            action = algo.compute_single_action(observation=obs_processed, prev_action=action, prev_reward=reward)
            obs, reward, terminated, truncated, _ = env.step(action)
            print(f"Action: {Actions(action).name}, Reward: {reward}, Done: {terminated}")
            env.render()
    env.close()
    print("Done!")
