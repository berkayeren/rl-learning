import faulthandler

import numpy as np
import ray
import torch
from gymnasium.wrappers import ResizeObservation
from minigrid.core.actions import Actions
from minigrid.manual_control import ManualControl
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, RGBImgObsWrapper
from ray.rllib.algorithms import PPOConfig, PPO
from ray.tune import register_env

from environments.empty import EmptyEnv
from environments.minigrid_wrapper import PositionBasedWrapper
from trainer import CustomEnv, plot_heatmap

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, configure_logging=False)
    faulthandler.disable()

    env = CustomEnv(size=19, render_mode="human", highlight=True, env_type=3, enable_dowham_reward_v2=True)
    # # Register the custom environment
    # register_env("CustomPlaygroundCrossingEnv-v0",
    #              lambda config:
    #              PositionBasedWrapper(
    #                  CustomEnv(**config)))

    # algo = PPO.from_checkpoint(
    #     r"/Users/berkayeren/ray_results/PPO_Multiroom_dv2_32_32/DoWhaMV2_multi_room_batch1024[32, 32]5.0_6_env_config=env_type_3_max_steps_1444_conv_filter_False_enable_dowham_reward_v1_False_en_2025-07-14_22-23-26/checkpoint_001495",
    #     config=PPOConfig().env_runners(
    #         num_env_runners=1,
    #         num_envs_per_env_runner=1,
    #     ).environment(env="CustomPlaygroundCrossingEnv-v0"))

    algo = None
    manual_control = ManualControl(env, seed=42)
    manual_control.start()
    states = []
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
        states.append(env.env.states)

    env.close()

    # After collecting states for all episodes, sum them into one heatmap
    import numpy as np  # ensure numpy is imported for summation

    summed_states = np.sum(np.array(states), axis=0)
    from types import SimpleNamespace

    # Create a dummy env object matching plot_heatmap signature
    base = env.env  # underlying CustomEnv instance
    dummy_env = SimpleNamespace(
        states=summed_states,
        width=base.width,
        height=base.height,
        grid=getattr(base, 'grid', None),
        goal_pos=getattr(base, 'goal_pos', None)
    )
    plot_heatmap(dummy_env, f"heatmaps/heat_map_sum.png")

    print("Done!")
