import faulthandler

import numpy as np
import ray
import torch
from gymnasium.wrappers import ResizeObservation
from minigrid.core.actions import Actions
from minigrid.envs import PlaygroundEnv, ObstructedMaze_Full, DoorKeyEnv
from minigrid.envs.babyai import OpenDoor
from minigrid.manual_control import ManualControl
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper
from ray.rllib.algorithms import PPOConfig, PPO
from ray.tune import register_env

from environments.empty import EmptyEnv
import gymnasium as gym
from environments.minigrid_wrapper import PositionBasedWrapper
from trainer import CustomEnv, plot_heatmap

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, configure_logging=False)
    faulthandler.disable()

    env = ImgObsWrapper(RGBImgPartialObsWrapper(
        OpenDoor(highlight=True, max_steps=1444, render_mode="human", select_by="loc"),
        tile_size=12))
    # # Register the custom environment
    register_env("CustomPlaygroundCrossingEnv-v0",
                 lambda config:

                 ImgObsWrapper(RGBImgPartialObsWrapper(CustomEnv(**config), tile_size=12)))

    algo = PPO.from_checkpoint(
        r"/Users/berkayeren/Library/Mobile Documents/com~apple~CloudDocs/Thesis/training results/PPO_TwelveRoom_4Room_Dv2ScheduledEntrFinal/DoWhaMV2_twelve_rooms_batch4000[512, 512]DowhamTrue_0_env_config=env_type_Environments_twelve_rooms_4_max_steps_1444_conv_filter_F_2025-10-17_12-27-03/checkpoint_000112",
        # r"/Users/berkayeren/Library/Mobile Documents/com~apple~CloudDocs/Thesis/training results/PPO_TwelveRoom_4Room_Dv1ScheduledEntrF/dh1_twe_da2b3_00006/checkpoint_000110",
        config=PPOConfig().env_runners(
            num_env_runners=1,
            num_envs_per_env_runner=1,
        ).environment(env="CustomPlaygroundCrossingEnv-v0").rl_module(
            model_config_dict={
                "use_lstm": True,
                "lstm_cell_size": 256,
            }
        ))

    algo = None
    manual_control = ManualControl(env, seed=42)
    manual_control.start()
    states = []
    for episode in range(0, 100):
        print("Episode:", episode)
        action = 0
        reward = 0
        terminated = False
        truncated = False
        obs, _ = env.reset()
        env.states = np.full((env.env.env.width, env.env.env.height), 0)
        # Initialize LSTM state for the episode
        lstm_state = algo.get_policy().get_initial_state()

        while not (terminated or truncated):
            action, lstm_state, _ = algo.compute_single_action(
                observation=obs,
                state=lstm_state,
                prev_action=action,
                prev_reward=reward
            )
            env.states[env.env.env.agent_pos[0]][env.env.env.agent_pos[1]] += 1
            obs, reward, terminated, truncated, _ = env.step(action)
            # print(f"Action: {Actions(action).name}, Reward: {reward}, Done: {terminated}")
            env.render()
        states.append(env.states)

    env.close()

    # After collecting states for all episodes, sum them into one heatmap
    import numpy as np  # ensure numpy is imported for summation

    summed_states = np.sum(np.array(states), axis=0)
    from types import SimpleNamespace

    # Create a dummy env object matching plot_heatmap signature
    base = env.env.env  # underlying CustomEnv instance
    dummy_env = SimpleNamespace(
        states=summed_states,
        width=base.width,
        height=base.height,
        grid=getattr(base, 'grid', None),
        goal_pos=getattr(base, 'goal_pos', None)
    )
    plot_heatmap(dummy_env, f"heat_map_sum_obstructed_dv2.png")

    print("Done!")
