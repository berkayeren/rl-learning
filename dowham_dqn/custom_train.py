import argparse
import datetime
import os
import sys
from typing import Optional, Union, Dict

import gym
import numpy as np
import ray
import torch
import torch.nn as nn
from minigrid.wrappers import ImgObsWrapper
from ray.experimental.tqdm_ray import tqdm
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module import RLModuleSpec, RLModule
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import PolicyID, EpisodeType
from ray.tune import register_env
from torch import optim

from custom_dqn_model import NatureCNN
from custom_playground_env import CustomPlaygroundEnv, MiniGridNet

# Suppress Deprecation Warnings and Ray duplicate logs
os.environ['PYTHONWARNINGS'] = "ignore::DeprecationWarning"
os.environ["RAY_DEDUP_LOGS"] = "0"

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Register the custom model
ModelCatalog.register_custom_model("MinigridPolicyNet", NatureCNN)

# Argument parser setup
parser = argparse.ArgumentParser(description="Custom training script")
parser.add_argument('--render_mode', type=str, help='The render mode parameter', default=None)
parser.add_argument('--num_env_runners', type=int, help='Number of environment runners (was num_rollout_workers)',
                    default=1)
parser.add_argument('--num_envs_per_env_runner', type=int,
                    help='Number of environments per env runner (was num_envs_per_worker)', default=1)
parser.add_argument('--num_gpus', type=int, help='The number of GPUs', default=0)
parser.add_argument('--algo', type=str, help='The algorithm to use (dqn or ppo)', default='dqn')
parser.add_argument('--start', type=int, help='Start Index', default=0)
parser.add_argument('--end', type=int, help='End Index', default=50000)
parser.add_argument('--restore', action='store_true', help='Restore from checkpoint')
parser.add_argument('--checkpoint_path', type=str, help='Checkpoint Path', default='')
parser.add_argument('--enable_prediction_reward', action='store_true', help='Enable prediction reward')
parser.add_argument('--enable_dowham_reward', action='store_true', help='Enable prediction reward')
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
        self.visited_states = None
        self.states = None
        self.height = 0
        self.width = 0
        self.path = path

    def on_episode_start(
            self,
            *,
            episode: Union[EpisodeType, EpisodeV2], env_runner: Optional["EnvRunner"] = None,
            metrics_logger: Optional[MetricsLogger] = None,
            env: Optional[gym.Env] = None,
            env_index: int,
            rl_module: Optional[RLModule] = None,
            worker: Optional["EnvRunner"] = None,
            base_env: Optional[BaseEnv] = None,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            **kwargs,
    ) -> None:
        self.visited_states = set()
        self.height = env.envs[env_index].height
        self.width = env.envs[env_index].width
        self.states = np.full((self.width, self.height), 0)

    def on_episode_step(
            self,
            *,
            episode: SingleAgentEpisode, env_runner: Optional["EnvRunner"] = None,
            metrics_logger: Optional[MetricsLogger] = None,
            env: Optional[gym.Env] = None,
            env_index: int,
            rl_module: Optional[RLModule] = None,
            worker: Optional["EnvRunner"] = None,
            base_env: Optional[BaseEnv] = None,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            **kwargs,
    ) -> None:
        env = env.envs[env_index].unwrapped
        x, y = env.agent_pos
        self.states[x][y] += 1

        ## if hasattr(env, "intrinsic_reward"):
        ##     episode.custom_metrics["intrinsic_reward"] = env.intrinsic_reward

    ##
    ## if hasattr(env, "count_bonus"):
    ##     episode.custom_metrics["count_bonus"] = env.count_bonus
    ##
    ## if hasattr(env, "enable_prediction_reward") and env.enable_prediction_reward:
    ##     episode.custom_metrics["prediction_reward"] = env.prediction_reward
    ##     episode.custom_metrics["prediction_prob"] = env.prediction_prob

    def preprocess_observation(self, obs):
        image = torch.tensor(obs["image"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        direction = torch.tensor([[obs["direction"]]], dtype=torch.float32)
        return image, direction

    def on_episode_end(
            self,
            *,
            episode: Union[EpisodeType, EpisodeV2], env_runner: Optional["EnvRunner"] = None,
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
        env = env.envs[env_index].unwrapped
        if env.enable_prediction_reward:
            for eh in env.episode_history:
                current_obs = eh["current_obs"]
                action = eh["action"]
                image, direction = self.preprocess_observation(current_obs)
                env.prediction_optimizer.zero_grad()
                outputs = env.prediction_net(image, direction)
                loss = env.prediction_criterion(outputs, torch.tensor([action], dtype=torch.long))
                loss.backward()
                env.prediction_optimizer.step()
            # Save the prediction network
            torch.save({
                'model_state_dict': env.prediction_net.state_dict(),
                'optimizer_state_dict': env.prediction_optimizer.state_dict(),
            },
                f'{os.path.join(self.path, "prediction_network")}/prediction_network_checkpoint.pth')

        total_size = self.width * self.height
        unique_states_visited = np.count_nonzero(self.states)
        percentage_visited = (unique_states_visited / total_size) * 100
        # episode.custom_metrics["percentage_visited"] = percentage_visited
        # episode.custom_metrics["left"] = env.action_count[0]
        # episode.custom_metrics["right"] = env.action_count[1]
        # episode.custom_metrics["forward"] = env.action_count[2]
        # episode.custom_metrics["pickup"] = env.action_count[3]
        # episode.custom_metrics["drop"] = env.action_count[4]
        # episode.custom_metrics["toggle"] = env.action_count[5]
        # episode.custom_metrics["done"] = env.action_count[6]


def get_trainer_config(algo_name, args, net, criterion, optimizer, total_cpus, output_folder_path, formatted_time):
    if algo_name.lower() == 'dqn':
        from ray.rllib.algorithms.dqn import DQNConfig
        config = (
            DQNConfig()
            .environment(env="MiniGrid-CustomPlayground-v0")
            .env_runners(
                num_env_runners=args.num_env_runners,
                num_envs_per_env_runner=args.num_envs_per_env_runner,
                num_cpus_per_env_runner=total_cpus / args.num_env_runners,
                num_gpus_per_env_runner=args.num_gpus / args.num_env_runners,
                sample_timeout_s=60,  # Increase sample timeout
                rollout_fragment_length=50  # Decrease rollout fragment length
            )
            .callbacks(AccuracyCallback)
            .training(
                lr=1e-5,  # Learning rate
                gamma=0.99,  # Discount factor
                train_batch_size=args.batch_size,  # Batch size
                num_atoms=1,
                v_min=-10.0,
                v_max=10.0,
                noisy=False,
                dueling=True,  # Use dueling architecture
                double_q=True,  # Use double Q-learning
                n_step=3,  # N-step Q-learning
                target_network_update_freq=500,
            )
            .rl_module(
                # Plug-in our custom RLModule class.
                rl_module_spec=RLModuleSpec(
                    module_class=NatureCNN,
                ),
            )
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .resources(
                num_gpus=args.num_gpus,
            ).reporting(
                keep_per_episode_custom_metrics=True,
            )
            .framework("torch")
            .fault_tolerance(
                recreate_failed_env_runners=True,
                restart_failed_sub_environments=True
            )
        )
    elif algo_name.lower() == 'ppo':
        from ray.rllib.algorithms.ppo import PPOConfig
        config = (
            PPOConfig()
            .environment(env="MiniGrid-CustomPlayground-v0")
            .env_runners(
                num_env_runners=args.num_env_runners,
                num_envs_per_env_runner=args.num_envs_per_env_runner,
                num_cpus_per_env_runner=total_cpus / args.num_env_runners,
                num_gpus_per_env_runner=args.num_gpus / args.num_env_runners,
            )
            .callbacks(AccuracyCallback)
            .training(
                lr=1e-5,  # Learning rate
                model={
                    "custom_model": "MinigridPolicyNet",
                },
                gamma=0.99,  # Discount factor
                train_batch_size=args.batch_size,  # Batch size
                minibatch_size=args.batch_size,  # Batch size
                use_gae=True,
                lambda_=0.95,
                clip_param=0.2,
                vf_clip_param=10.0,
                entropy_coeff=0.01,
            )
            .resources(
                num_gpus=args.num_gpus,
            ).reporting(
                keep_per_episode_custom_metrics=True,
            )
            .framework("torch")
            .fault_tolerance(
                recreate_failed_env_runners=True,
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
    # Get current date and time
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H-%M")

    # Create output directories
    os.makedirs(output_folder_path, exist_ok=True)
    os.makedirs(os.path.join(output_folder_path, "prediction_network"), exist_ok=True)
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
                     prediction_net=net,
                     prediction_criterion=criterion,
                     prediction_optimizer=optimizer,
                     enable_prediction_reward=args.enable_prediction_reward,
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
    for i in tqdm(range(args.start, args.end + 1)):
        result = trainer.train()

        if i % args.checkpoint_size == 0:
            # Save the model checkpoint
            checkpoint_dir = os.path.join(output_folder_path, "checkpoint")
            checkpoint_path = trainer.save(checkpoint_dir)
            print(f"Checkpoint saved at iteration {i} to {checkpoint_path}")

    # Shutdown Ray
    ray.shutdown()
