import faulthandler

import numpy as np
import ray
import torch
from gymnasium.wrappers import ResizeObservation
from minigrid.core.actions import Actions
from minigrid.manual_control import ManualControl
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper
from ray.rllib.algorithms import PPOConfig, PPO
from ray.tune import register_env

from environments.empty import EmptyEnv
from environments.minigrid_wrapper import PositionBasedWrapper
from trainer import CustomEnv, plot_heatmap

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, configure_logging=False)
    faulthandler.disable()

    env = ImgObsWrapper(RGBImgPartialObsWrapper(
        CustomEnv(size=19, render_mode="human", highlight=True, env_type=4, enable_dowham_reward_v2=True),
        tile_size=12))
    # # Register the custom environment
    register_env("CustomPlaygroundCrossingEnv-v0",
                 lambda config:

                 ImgObsWrapper(RGBImgPartialObsWrapper(CustomEnv(**config), tile_size=12)))

    algo = PPO.from_checkpoint(
        r"/Users/berkayeren/ray_results/PPO_TwelveRoom/DoWhaMV2_twelve_rooms_batch4000[512, 512]DowhamTrue_1_env_config=env_type_Environments_twelve_rooms_4_max_steps_1444_conv_filter_F_2025-09-13_13-56-23/checkpoint_000024",
        config=PPOConfig().env_runners(
            num_env_runners=1,
            num_envs_per_env_runner=1,
        ).environment(env="CustomPlaygroundCrossingEnv-v0").rl_module(
            model_config_dict={
                "use_lstm": True,
                "lstm_cell_size": 256,
            }
        ))

    # algo = None
    # manual_control = ManualControl(env, seed=42)
    # manual_control.start()
    states = []
    for episode in range(0, 10):
        action = 0
        reward = 0
        terminated = False
        truncated = False
        obs, _ = env.reset()
        # Initialize LSTM state for the episode
        lstm_state = algo.get_policy().get_initial_state()

        while not (terminated or truncated):
            action, lstm_state, _ = algo.compute_single_action(
                observation=obs,
                state=lstm_state,
                prev_action=action,
                prev_reward=reward
            )
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
