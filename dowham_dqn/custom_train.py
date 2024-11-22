import argparse
import datetime
import os
import sys
from collections import deque
from typing import Optional, Union, Dict

import numpy as np
import ray
import torch
import torch.nn as nn
from minigrid.wrappers import ImgObsWrapper
from ray.experimental.tqdm_ray import tqdm
from ray.rllib import BaseEnv, Policy, SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import PolicyID
from ray.tune import register_env
from torch import optim

from custom_dqn_model import NatureCNN
from custom_playground_env import CustomPlaygroundEnv, MiniGridNet

# Suppress Deprecation Warnings and Ray duplicate logs
os.environ['PYTHONWARNINGS'] = "ignore::DeprecationWarning"
os.environ["RAY_DEDUP_LOGS"] = "0"

# Register the custom model
ModelCatalog.register_custom_model("MinigridPolicyNet", NatureCNN)

# Argument parser setup
parser = argparse.ArgumentParser(description="Custom training script")
parser.add_argument('--render_mode', type=str, help='The render mode parameter', default=None)
parser.add_argument('--num_rollout_workers', type=int, help='The number of rollout workers', default=1)
parser.add_argument('--num_envs_per_worker', type=int, help='The number of environments per worker', default=1)
parser.add_argument('--num_gpus', type=int, help='The number of GPUs', default=0)
parser.add_argument('--algo', type=str, help='The algorithm to use (dqn or ppo)', default='dqn')
parser.add_argument('--start', type=int, help='Start Index', default=0)
parser.add_argument('--end', type=int, help='End Index', default=50000)
parser.add_argument('--restore', action='store_true', help='Restore from checkpoint')
parser.add_argument('--checkpoint_path', type=str, help='Checkpoint Path', default='')
parser.add_argument('--enable_dowham_reward', action='store_true', help='Enable DoWham Intrinsic reward')
parser.add_argument('--output_folder', type=str, help='Output Folder', default="ray_results")
parser.add_argument('--batch_size', type=int, help='Batch Size', default=32)
parser.add_argument('--checkpoint_size', type=int, help='Iteration Number to take checkpoint', default=100000)
args = parser.parse_args()

# Set up output folder path
current_dir = os.getcwd()

# Append the output folder to the current file path
output_folder_path = os.path.join(os.path.dirname(current_dir), args.output_folder)


class AccuracyCallback(DefaultCallbacks):
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
            episode: Union["Episode", "EpisodeV2"],
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        self.visited_states = set()
        self.height = base_env.get_sub_environments()[env_index].unwrapped.height
        self.width = base_env.get_sub_environments()[env_index].unwrapped.width
        self.states = np.full((self.width, self.height), 0)

    def on_episode_step(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            episode: "EpisodeV2",
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        env = base_env.get_sub_environments()[env_index].unwrapped

        episode.custom_metrics["intrinsic_reward"] = env.intrinsic_reward
        episode.custom_metrics["step_done"] = env.done

    def on_episode_end(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: EpisodeV2,
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        env = base_env.get_sub_environments()[env_index].unwrapped
        episode.custom_metrics["left"] = env.action_count[0]
        episode.custom_metrics["right"] = env.action_count[1]
        episode.custom_metrics["forward"] = env.action_count[2]
        episode.custom_metrics["pickup"] = env.action_count[3]
        episode.custom_metrics["drop"] = env.action_count[4]
        episode.custom_metrics["toggle"] = env.action_count[5]
        episode.custom_metrics["done"] = env.action_count[6]
        episode.custom_metrics["success_rate"] = env.success_rate
        episode.custom_metrics["success_history_len"] = len(env.success_history)
        # print(
        #     f"Reward:{episode.total_reward} | env.success_rate:{env.success_rate} | Len:{len(env.success_history)} | env.minNumRooms:{env.minNumRooms}")

    # def on_learn_on_batch(
    #        self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    # ) -> None:
    #    seq_lens = train_batch.get("seq_lens", 16)
    #    train_batch['seq_lens'] = np.full_like(seq_lens, 16)
    #    super().on_learn_on_batch(policy=policy, train_batch=train_batch, result=result, **kwargs)


def get_trainer_config(algo_name, args, net, criterion, optimizer, total_cpus, output_folder_path, formatted_time):
    if algo_name.lower() == 'dqn':
        from ray.rllib.algorithms.dqn import DQNConfig
        config = (
            DQNConfig()
            .environment(env="MiniGrid-CustomPlayground-v0")
            .rollouts(
                num_rollout_workers=args.num_rollout_workers,
                num_envs_per_worker=args.num_envs_per_worker,
                batch_mode="truncate_episodes",  # Necessary for RNNs
            )
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
                lr=0.00025,
                optimizer={"type": "Adam"},
                model={
                    "conv_filters": [
                        [32, [3, 3], 2],
                        [64, [3, 3], 2],
                        [128, [3, 3], 2],
                        [256, [1, 1], 1],
                    ],
                    "conv_activation": "relu",
                    "fcnet_hiddens": [512],
                    "fcnet_activation": "relu",
                    "use_lstm": False,
                    "lstm_cell_size": 256,
                    "max_seq_len": 16,
                    "lstm_use_prev_reward": True,
                    "lstm_use_prev_action": True,
                },
                gamma=0.99,
                train_batch_size=32,
                dueling=True,
                double_q=True,
                n_step=3,
                target_network_update_freq=500,
                replay_buffer_config={
                    "type": "ReplayBuffer",
                    "capacity": 50000,  # Replay buffer capacity
                    "replay_sequence_length": 16,  # Ensure sequence handling
                    "seq_lens": 16,  # Ensure sequence handling
                }
            )
            .resources(
                num_gpus=args.num_gpus,
                num_cpus_per_worker=total_cpus / args.num_rollout_workers,
                num_gpus_per_worker=args.num_gpus / args.num_rollout_workers,
            )
            .framework("torch")
            .fault_tolerance(
                recreate_failed_workers=True,
                restart_failed_sub_environments=True
            )
        )
    elif algo_name.lower() == 'ppo':
        from ray.rllib.algorithms.ppo import PPOConfig
        config = (
            PPOConfig()
            .environment(env="MiniGrid-CustomPlayground-v0")
            .rollouts(
                num_rollout_workers=args.num_rollout_workers,
                num_envs_per_worker=args.num_envs_per_worker
            )
            .callbacks(AccuracyCallback)
            .training(
                model={
                    "conv_filters": [
                        [32, [3, 3], 2],  # 1st layer: 32 filters, 3x3 kernel, stride 2
                        [64, [3, 3], 2],  # 2nd layer: 64 filters, 3x3 kernel, stride 2
                        [128, [3, 3], 2],  # 3rd layer: 128 filters, 3x3 kernel, stride 2
                        [256, [1, 1], 1],  # 4th layer: 256 filters, 1x1 kernel, stride 1
                    ],
                    "conv_activation": "relu",  # Activation function
                    "fcnet_hiddens": [512],  # Fully connected layers with 512 units
                    "fcnet_activation": "relu",  # Activation function for fully connected layers
                    "vf_share_layers": True,  # share layers between actor and critic
                    "use_lstm": True,  # Enable LSTM
                    "lstm_cell_size": 256,  # Size of the LSTM cell
                    "max_seq_len": 16,  # Maximum sequence length
                    # Optional: Include previous actions and rewards in the LSTM input
                    "lstm_use_prev_reward": True,
                    "lstm_use_prev_action": True,
                },
                gamma=0.99,  # Discount factor
                lr=0.00025,  # Learning rate from the best config
                train_batch_size=32,  # Batch size
                sgd_minibatch_size=16,  # Size of SGD minibatches
                num_sgd_iter=10,  # Number of SGD iterations per epoch
                clip_param=0.2,  # PPO clip parameter
                entropy_coeff=0.05,  # Entropy regularization coefficient
                use_gae=True,  # Use Generalized Advantage Estimation
            )
            .resources(
                num_gpus=args.num_gpus,
                # num_cpus_per_worker=total_cpus / args.num_rollout_workers,
                # num_gpus_per_worker=args.num_gpus / args.num_rollout_workers,
            )
            .framework("torch")
            .fault_tolerance(
                recreate_failed_workers=True,
                restart_failed_sub_environments=True
            )
        )
    else:
        raise ValueError(f"Unknown algorithm specified: {algo_name}")

    # Set up logger config
    config = config.to_dict()
    config['logger_config'] = {
        "type": "ray.tune.logger.UnifiedLogger",
        "logdir": os.path.join(output_folder_path, f'results/result_{formatted_time}'),
    }
    # Convert back to config class
    if algo_name.lower() == 'dqn':
        config = DQNConfig.from_dict(config)
    elif algo_name.lower() == 'ppo':
        config = PPOConfig.from_dict(config)
    return config


if __name__ == "__main__":
    # Initialize Ray
    ray.init(ignore_reinit_error=True, num_gpus=args.num_gpus)

    # Get current date and time
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H-%M")

    # Create output directories
    os.makedirs(output_folder_path, exist_ok=True)
    os.makedirs(os.path.join(output_folder_path, "results"), exist_ok=True)
    os.makedirs(os.path.join(output_folder_path, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(output_folder_path, "results", f"result_{formatted_time}"), exist_ok=True)

    # Initialize net, criterion, optimizer
    net = MiniGridNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Register the custom environment
    register_env("MiniGrid-CustomPlayground-v0",
                 lambda config: ImgObsWrapper(CustomPlaygroundEnv(
                     render_mode=args.render_mode,
                     enable_dowham_reward=args.enable_dowham_reward,
                 )))

    # Get total CPUs
    total_cpus = os.cpu_count()
    print(f"Total number of available CPUs: {total_cpus}")

    # Get trainer configuration based on algorithm
    config = get_trainer_config(
        args.algo,
        args,
        net,
        criterion,
        optimizer,
        total_cpus,
        output_folder_path,
        formatted_time
    )

    # Build the trainer
    trainer = config.build()

    # Restore from checkpoint if needed
    if args.restore and args.checkpoint_path:
        try:
            trainer.restore(args.checkpoint_path)
        except ValueError:
            sys.stdout.write("Checkpoint not found, starting from scratch.\n")

    # Training loop
    for i in range(args.start, args.end + 1):
        result = trainer.train()

        if i % args.checkpoint_size == 0:
            # Save the model checkpoint
            checkpoint_dir = os.path.join(output_folder_path, "checkpoint")
            checkpoint_path = trainer.save(f'{checkpoint_dir}/checkpoint-{i}')
            print(f"Checkpoint saved at iteration {i} to {checkpoint_path}")

    # Shutdown Ray
    ray.shutdown()
