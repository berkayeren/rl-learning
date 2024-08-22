import argparse
import datetime
import logging
import os
import sys
from typing import Optional, Union, Dict

import numpy as np
import ray
import torch
import torch.nn as nn
from minigrid.wrappers import ImgObsWrapper
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import PolicyID
from ray.tune import register_env
from torch import optim
from tqdm import tqdm

from custom_dqn_model import MinigridPolicyNet
from custom_playground_env import CustomPlaygroundEnv, MiniGridNet

os.environ['PYTHONWARNINGS'] = "ignore::DeprecationWarning"
# Initialize Ray
ray.init(ignore_reinit_error=True, _metrics_export_port=8080)

# Register the custom model
ModelCatalog.register_custom_model("MinigridPolicyNet", MinigridPolicyNet)

# Create the parser
parser = argparse.ArgumentParser(description="Get the render mode parameter")

# Add the arguments
parser.add_argument('--render_mode', type=str, help='The render mode parameter', default=None)
parser.add_argument('--num_rollout_workers', type=int, help='The number of rollout workers', default=1)
parser.add_argument('--num_envs_per_worker', type=int, help='The number of environments per worker', default=1)
parser.add_argument('--num_gpus', type=int, help='The number of environments per worker', default=0)
parser.add_argument('--algo', type=int, help='The algorithm to use', default=0)
parser.add_argument('--start', type=int, help='Start Index', default=0)
parser.add_argument('--end', type=int, help='End Index', default=50000)
parser.add_argument('--restore', type=bool, help='Restore from checkpoint', default=True)
parser.add_argument('--enable_prediction_reward', type=bool, help='Restore from checkpoint', default=False)
parser.add_argument('--output_folder', type=str, help='Output Folder', default="ray_results")
parser.add_argument('--batch_size', type=int, help='Batch Size', default=32)

# Parse the arguments
args = parser.parse_args()

# Get the current working directory
current_dir = os.getcwd()

# Append the output folder to the current file path
output_folder_path = os.path.join(os.path.dirname(current_dir), args.output_folder)


class AccuracyCallback(DefaultCallbacks):
    training_iteration = 0  # Class-level attribute

    def __init__(self, path=output_folder_path):
        super().__init__()
        self.states = None
        self.height = 0
        self.width = 0
        self.path = path

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
        env = base_env.get_sub_environments()[0].unwrapped
        x, y = base_env.get_sub_environments()[0].unwrapped.agent_pos
        self.states[x][y] += 1

        if hasattr(env, "dowham_reward") and hasattr(env, "intrinsic_reward"):
            episode.custom_metrics["intrinsic_reward"] = env.intrinsic_reward

        if hasattr(env, "count_exploration") and hasattr(env, "count_bonus"):
            episode.custom_metrics["count_bonus"] = env.count_bonus

        episode.custom_metrics["prediction_reward"] = env.prediction_reward
        episode.custom_metrics["prediction_prob"] = env.prediction_prob

    def preprocess_observation(self, obs):
        image = torch.tensor(obs["image"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        direction = torch.tensor([[obs["direction"]]], dtype=torch.float32)
        return image, direction

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
        env = base_env.get_sub_environments()[0].unwrapped
        if env.enable_prediction_reward:
            for env in base_env.get_sub_environments():
                env = env.unwrapped
                # Iterate through all collected episodes
                for eh in env.episode_history:
                    # Extract the current observation from the episode
                    current_obs = eh["current_obs"]
                    # Extract the action taken in the episode
                    action = eh["action"]

                    # Preprocess the current observation to get the image and direction tensors
                    image, direction = self.preprocess_observation(current_obs)

                    # Zero the gradients of the prediction optimizer
                    env.prediction_optimizer.zero_grad()
                    # Perform a forward pass through the prediction network
                    outputs = env.prediction_net(image, direction)
                    # Compute the loss between the network's output and the actual action taken
                    loss = env.prediction_criterion(outputs, torch.tensor([action], dtype=torch.long))
                    # Perform a backward pass to compute the gradients
                    loss.backward()
                    # Update the model parameters using the computed gradients
                    env.prediction_optimizer.step()
            # TODO: Save model every n episodes
            torch.save({
                'model_state_dict': env.prediction_net.state_dict(),
                'optimizer_state_dict': env.prediction_optimizer.state_dict(),
            },
                f'{os.path.join(self.path, "prediction_network")}/prediction_network_checkpoint.pth')

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
    # Get the current date and time
    now = datetime.datetime.now()

    # Format the date and time
    formatted_time = now.strftime("%Y-%m-%d_%H-%M")

    # Create the folder at output_folder_path if it does not exist
    os.makedirs(output_folder_path, exist_ok=True)
    os.makedirs(os.path.join(output_folder_path, "prediction_network"), exist_ok=True)
    os.makedirs(os.path.join(output_folder_path, "results"), exist_ok=True)
    os.makedirs(os.path.join(output_folder_path, "checkpoint"), exist_ok=True)

    algo = {
        0: {"enable_dowham_reward": True},
        1: {"enable_count_based": True},
        2: {"enable_count_based": False, "enable_dowham_reward": False},
    }

    # Get the parameters
    render_mode = args.render_mode
    num_rollout_workers = args.num_rollout_workers
    num_envs_per_worker = args.num_envs_per_worker
    num_gpus = args.num_gpus
    enable_prediction_reward = args.enable_prediction_reward
    batch_size = args.batch_size
    net = MiniGridNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Register the custom environment
    register_env("MiniGrid-CustomPlayground-v0",
                 lambda config: ImgObsWrapper(CustomPlaygroundEnv(render_mode=render_mode,
                                                                  prediction_net=net,
                                                                  prediction_criterion=criterion,
                                                                  prediction_optimizer=optimizer,
                                                                  enable_prediction_reward=enable_prediction_reward,
                                                                  **algo[args.algo])))

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
            train_batch_size=batch_size,  # Batch size
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
        ).framework("torch").fault_tolerance(recreate_failed_workers=True,
                                             restart_failed_sub_environments=True))

    config = config.to_dict()
    config['logger_config'] = {
        "type": "ray.tune.logger.UnifiedLogger",  # This is the default logger used
        "logdir": os.path.join(output_folder_path, f'results/result_{formatted_time}'),
    }

    # Convert the dictionary back to DQNConfig
    config = DQNConfig.from_dict(config)

    # Initialize the DQN Trainer with the configured settings
    dqn_trainer = config.build()

    # Define the relative path to the directory where you want to save the model
    relative_path = "checkpoint"

    # Join the current directory with the relative path
    checkpoint_dir = os.path.join(output_folder_path, relative_path)

    # Configure the logging
    logging.basicConfig(
        filename=f'{output_folder_path}/minigrid.log',  # Log file name
        level=logging.DEBUG,  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
        datefmt='%Y-%m-%d %H:%M:%S'  # Date format
    )

    if args.restore:
        try:
            dqn_trainer.restore(f'{checkpoint_dir}/checkpoint-algo{args.algo}')
        except ValueError:
            sys.stdout.write("Checkpoint not found, starting from scratch.\n")

    # Training loop
    for i in tqdm(range(args.start, args.end + 1)):  # Number of training iterations
        result = dqn_trainer.train()
        result["iteration"] = i

        logging.info(result)

        if i % 100000 == 0:
            # Save the model checkpoint
            checkpoint = dqn_trainer.save(f'{checkpoint_dir}/checkpoint-algo{args.algo}-{i}')

    # Shutdown Ray
    ray.shutdown()
