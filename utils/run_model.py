import argparse
import os
import sys
from datetime import datetime
from typing import Optional, Union, Dict

import numpy as np
import pandas as pd
import ray
from custom_dqn_model import MinigridPolicyNet
from custom_playground_env import CustomPlaygroundEnv
from minigrid.wrappers import ImgObsWrapper
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms import DQN
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import PolicyID
from ray.tune import register_env

# Initialize Ray
ray.init(ignore_reinit_error=True, _metrics_export_port=8080)

# Register the custom model
ModelCatalog.register_custom_model("MinigridPolicyNet", MinigridPolicyNet)


class AccuracyCallback(DefaultCallbacks):
    """
    A custom callback class that extends the DefaultCallbacks class from Ray RLlib.
    This class is used to track the agent's performance and exploration in the environment.
    """

    def __init__(self):
        """
        Initialize the AccuracyCallback class.
        """
        super().__init__()
        self.states = None
        self.height = 0
        self.width = 0

    def on_episode_start(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Union[Episode, EpisodeV2],
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        """
        This method is called at the start of each episode.
        It initializes the visited states and the environment dimensions.
        """
        super().on_episode_start(worker=worker, base_env=base_env, policies=policies, episode=episode,
                                 env_index=env_index, **kwargs)
        self.visited_states = set()
        self.height = base_env.get_sub_environments()[0].unwrapped.height
        self.width = base_env.get_sub_environments()[0].unwrapped.width
        self.states = np.full((self.width, self.height), 0)

    def on_episode_step(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            episode: EpisodeV2,
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        """
        This method is called at each step of the episode.
        It updates the visited states and the custom metrics for the episode.
        """
        env = base_env.get_sub_environments()[0].unwrapped
        x, y = base_env.get_sub_environments()[0].unwrapped.agent_pos
        self.states[x][y] += 1

        if hasattr(env, "dowham_reward") and hasattr(env, "intrinsic_reward"):
            episode.custom_metrics["intrinsic_reward"] = env.intrinsic_reward

        if hasattr(env, "count_exploration") and hasattr(env, "count_bonus"):
            episode.custom_metrics["count_bonus"] = env.count_bonus

    def on_episode_end(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Union[Episode, EpisodeV2, Exception],
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        """
        This method is called at the end of each episode.
        It calculates and logs the percentage of the environment visited by the agent.
        """
        env = base_env.get_sub_environments()[0].unwrapped
        total_size = self.width * self.height
        # Calculate the number of unique states visited by the agent
        unique_states_visited = np.count_nonzero(self.states)

        # Calculate the percentage of the environment the agent has visited
        percentage_visited = (unique_states_visited / total_size) * 100

        # Log the percentage
        episode.custom_metrics["percentage_visited"] = percentage_visited
        episode.custom_metrics["left"] = env.action_count[0]
        episode.custom_metrics["right"] = env.action_count[1]
        episode.custom_metrics["forward"] = env.action_count[2]
        episode.custom_metrics["pickup"] = env.action_count[3]
        episode.custom_metrics["drop"] = env.action_count[4]
        episode.custom_metrics["toggle"] = env.action_count[5]
        episode.custom_metrics["done"] = env.action_count[6]


if __name__ == "__main__":
    """
    The main function of the script.
    It sets up the environment, the DQN trainer, and the training loop.
    """

    # Create the parser
    parser = argparse.ArgumentParser(description="Get the render mode parameter")

    # Add the arguments
    parser.add_argument('--render_mode', type=str, help='The render mode parameter', default=None)
    parser.add_argument('--num_rollout_workers', type=int, help='The number of rollout workers', default=1)
    parser.add_argument('--num_envs_per_worker', type=int, help='The number of environments per worker', default=1)
    parser.add_argument('--num_gpus', type=int, help='The number of environments per worker', default=0)
    parser.add_argument('--algo', type=int, help='The algorithm to use', default=0)
    parser.add_argument('--checkpoint', type=str, help='Checkpoint Directory', default=None)

    algo = {
        0: {"enable_dowham_reward": True},
        1: {"enable_count_based": True},
        2: {"enable_count_based": False, "enable_dowham_reward": False},
    }

    # Parse the arguments
    args = parser.parse_args()

    # Get the parameters
    render_mode = args.render_mode
    num_rollout_workers = args.num_rollout_workers
    num_envs_per_worker = args.num_envs_per_worker
    num_gpus = args.num_gpus

    # Register the custom environment
    register_env("MiniGrid-CustomPlayground-v0",
                 lambda config: ImgObsWrapper(CustomPlaygroundEnv(render_mode=render_mode, **algo[args.algo])))

    # Define the DQN configuration
    config = (
        DQNConfig()
        .environment(env="MiniGrid-CustomPlayground-v0")
        .rollouts(num_rollout_workers=num_rollout_workers,
                  num_envs_per_worker=num_envs_per_worker)  # Adjust the number of workers as needed
        .exploration(
            explore=True,
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.1,
                "epsilon_timesteps": 10000,
            }
        )
        .callbacks(AccuracyCallback)
        .training(
            lr=1e-5,  # Learning rate
            optimizer={
                "type": "RMSProp",
                "lr": 1e-5,
                "weight_decay": 0,
                "momentum": 0,
                "centered": False
            },
            model={
                "custom_model": "MinigridPolicyNet",
            },
            gamma=0.99,  # Discount factor
            train_batch_size=32,  # Batch size
            num_atoms=1,
            v_min=-10.0,
            v_max=10.0,
            noisy=False,
            dueling=True,  # Use dueling architecture
            double_q=True,  # Use double Q-learning
            n_step=3,  # N-step Q-learning
            target_network_update_freq=500,
        )
        .resources(
            num_gpus=num_gpus,
            num_cpus_per_worker=1
        )
        .framework("torch").fault_tolerance(recreate_failed_workers=True, restart_failed_sub_environments=True)
        # .evaluation(
        #                 evaluation_parallel_to_training=False,
        #                 evaluation_sample_timeout_s=320,
        #                 evaluation_interval=10,
        #                 evaluation_duration=4,
        #                 evaluation_num_workers=0
        #             )
    )

    # Instantiate the DQN trainer
    dqn_trainer = DQN(config=config)

    # Get the current working directory
    current_dir = os.getcwd()

    # Define the relative path to the directory where you want to save the model
    relative_path = "checkpoint"

    # Join the current directory with the relative path
    checkpoint_dir = os.path.join(current_dir, relative_path)

    if args.checkpoint:
        try:
            dqn_trainer.restore(f'{checkpoint_dir}/{args.checkpoint}')
        except ValueError:
            sys.stdout.write("Checkpoint not found, starting from scratch.\n")

    env = ImgObsWrapper(CustomPlaygroundEnv(render_mode=None, **algo[args.algo]))

    for episode in range(0, 10):
        """
        Function to run a single episode.
        """
        sys.stdout.write(f"Running episode {episode + 1}\n")
        visited_states = {}
        done = False
        reward = 0
        action = None
        observation, _ = env.reset()

        while not done:

            # Compute the action using the trained policy
            action = dqn_trainer.compute_single_action(observation=observation, prev_action=action, prev_reward=reward)
            # Take the action in the environment
            observation, reward, done, info, _ = env.step(action)
            print(env.agent_pos)
            visited_states.setdefault(env.agent_pos, 0)
            visited_states[env.agent_pos] += 1
            if env.step_count >= 100000:
                done = True
            if render_mode == 'human':
                env.render()

        # Convert the dictionary to a DataFrame
        date_string = datetime.now().strftime("%Y-%m-%d-%H%M")
        df = pd.DataFrame(list(visited_states.items()))
        df.to_csv(f'trail-algo{args.algo}-episode{episode}-checkpoint{args.checkpoint}-{date_string}.csv', index=False)

    ray.shutdown()
