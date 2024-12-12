import argparse
import datetime
import os
import sys
from functools import partial
from typing import Union

import gymnasium as gym
import ray
from gymnasium.wrappers import ResizeObservation, TimeLimit
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from ray.experimental.tqdm_ray import tqdm
from ray.tune import register_env

from callbacks.minigrid.callback import MinigridCallback
from callbacks.pacman.callback import PacmanCallback
from environments.minigrid_env import CustomPlaygroundEnv
from environments.pacman_env import PacmanWrapper

os.environ['PYTHONWARNINGS'] = "ignore::DeprecationWarning"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ['PYTHONUNBUFFERED'] = "1"
os.environ['PYTHONWARNINGS'] = "ignore::DeprecationWarning"
os.environ['RAY_DEDUP_LOGS'] = "0"
os.environ['RAY_GRAFANA_HOST'] = "http://localhost:3000"
os.environ['RAY_GRAFANA_IFRAME_HOST'] = "http://localhost:3000"
os.environ['RAY_PROMETHEUS_HOST'] = "http://localhost:9090"

# Register the custom model
# ModelCatalog.register_custom_model("MinigridPolicyNet", NatureCNN)

# Argument parser setup
parser = argparse.ArgumentParser(description="Custom training script")
parser.add_argument('--render_mode', type=str, help='The render mode parameter', default=None)
parser.add_argument('--num_rollout_workers', type=int, help='The number of rollout workers', default=1)
parser.add_argument('--num_envs_per_worker', type=int, help='The number of environments per worker', default=1)
parser.add_argument('--num_gpus', type=int, help='The number of GPUs', default=0)
parser.add_argument('--algo', type=str, help='The algorithm to use (dqn or ppo)', default='dqn')
parser.add_argument('--env', type=str, help='The environment to use (minigrid or pacman)', default='minigrid')
parser.add_argument('--start', type=int, help='Start Index', default=0)
parser.add_argument('--end', type=int, help='End Index', default=50000)
parser.add_argument('--restore', action='store_true', help='Restore from checkpoint')
parser.add_argument('--checkpoint_path', type=str, help='Checkpoint Path', default='')
parser.add_argument('--enable_dowham_reward', action='store_true', help='Enable DoWham Intrinsic reward')
parser.add_argument('--output_folder', type=str, help='Output Folder', default="ray_results")
parser.add_argument('--batch_size', type=int, help='Batch Size', default=32)
parser.add_argument('--checkpoint_size', type=int, help='Iteration Number to take checkpoint', default=100000)
parser.add_argument('--use_tqdm', action='store_true', help='Use Tqdm')
args = parser.parse_args()

# Set up output folder path
current_dir = os.getcwd()

# Append the output folder to the current file path
output_folder_path = os.path.join(os.path.dirname(current_dir), args.output_folder)


def get_trainer_config(
        env_name: str,
        algo_name: str,
        args: argparse.Namespace,
        total_cpus: int,
        output_folder_path: str,
        formatted_time: str,
        callback: Union[MinigridCallback, PacmanCallback, None]
) -> Union["DQNConfig", "PPOConfig", "ImpalaConfig"]:
    """
    Generates the trainer configuration for the specified algorithm and environment.

    Args:
        env_name (str): The name of the environment to use.
        algo_name (str): The name of the algorithm to use (e.g., 'dqn' or 'ppo').
        args (argparse.Namespace): Parsed command-line arguments.
        total_cpus (int): The total number of CPUs available.
        output_folder_path (str): The path to the output folder for saving results and checkpoints.
        formatted_time (str): The formatted current date and time.
        callback (Union[MinigridCallback, PacmanCallback, None]): The callback class to use for custom metrics and logging.

    Returns:
        Union["DQNConfig", "PPOConfig"]: The configuration object for the specified algorithm.
    """

    if algo_name.lower() == 'dqn':
        from ray.rllib.algorithms.dqn import DQNConfig
        config = (
            DQNConfig()
            .environment(env=env_name, disable_env_checking=True)
            .rollouts(
                num_rollout_workers=args.num_rollout_workers,
                num_envs_per_worker=args.num_envs_per_worker,
                rollout_fragment_length=200 * args.num_envs_per_worker,
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
            ).evaluation(
                evaluation_parallel_to_training=False,
                evaluation_interval=100,
                evaluation_duration=10,
                evaluation_num_workers=0,
                evaluation_sample_timeout_s=60
            )
            .callbacks(partial(callback, path=output_folder_path))
            .training(
                gamma=0.99,  # Discount factor
                lr=1e-5,  # Learning rate
                train_batch_size=200 * args.num_envs_per_worker * args.num_rollout_workers,  # Batch size
                grad_clip=42,  # Max norm gradient
                optimizer={"type": "RMSProp"},
                model={
                    "dim": 88,
                    "conv_filters": [
                        [32, [3, 3], 5],  # Layer 1
                        [64, [3, 3], 5],  # Layer 2
                        [128, [3, 3], 2],  # Layer 3
                    ],
                    "conv_activation": "relu",
                    "fcnet_hiddens": [1024, 1024],
                    "post_fcnet_activation": "tanh"
                },
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
                num_cpus_per_worker=1,
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
            .environment(env=env_name, disable_env_checking=True)
            .rollouts(
                num_rollout_workers=args.num_rollout_workers,
                num_envs_per_worker=args.num_envs_per_worker
            )
            .callbacks(partial(callback, path=output_folder_path))
            .evaluation(
                evaluation_parallel_to_training=False,
                evaluation_interval=100,
                evaluation_duration=10,
                evaluation_num_workers=0,
                evaluation_sample_timeout_s=60
            )
            .training(
                model={
                    "dim": 88,
                    "conv_filters": [
                        [32, [3, 3], 5],  # Layer 1
                        [64, [3, 3], 5],  # Layer 2
                        [128, [3, 3], 2],  # Layer 3
                    ],
                    "conv_activation": "relu",
                    "fcnet_hiddens": [1024, 1024],
                    "post_fcnet_activation": "tanh",
                    "use_lstm": True,
                    "lstm_cell_size": 256,
                    "max_seq_len": 20,
                    "lstm_use_prev_action": True,
                    "lstm_use_prev_reward": True,
                    "vf_share_layers": False,
                    "post_fcnet_hiddens": [1024],
                },
                gamma=0.99,  # Discount factor
                lr=0.00025,  # Learning rate from the best config
                train_batch_size=4000,  # Batch size
                sgd_minibatch_size=64,  # Size of SGD minibatches
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
    elif algo_name.lower() == 'impala':
        from ray.rllib.algorithms import ImpalaConfig
        config = (
            ImpalaConfig()
            .environment(env=env_name, disable_env_checking=True)
            .rollouts(
                num_rollout_workers=args.num_rollout_workers,
                num_envs_per_worker=args.num_envs_per_worker,
                rollout_fragment_length=200 * args.num_envs_per_worker,
                batch_mode="truncate_episodes"
            )
            .evaluation(
                evaluation_parallel_to_training=False,
                evaluation_interval=100,
                evaluation_duration=10,
                evaluation_num_workers=0,
                evaluation_sample_timeout_s=60
            )
            .callbacks(partial(callback, path=output_folder_path))
            .training(
                gamma=0.99,  # Discount factor
                lr=1e-5,  # Learning rate
                train_batch_size=200 * args.num_envs_per_worker * args.num_rollout_workers,  # Batch size
                entropy_coeff=0.001,  # Entropy cost
                vf_loss_coeff=40,  # Baseline cost
                grad_clip=42,  # Max norm gradient
                optimizer={"type": "RMSProp"},
                model={
                    "dim": 88,
                    "conv_filters": [
                        [32, [3, 3], 5],  # Layer 1
                        [64, [3, 3], 5],  # Layer 2
                        [128, [3, 3], 2],  # Layer 3
                    ],
                    "conv_activation": "relu",
                    "fcnet_hiddens": [1024, 1024],
                    "post_fcnet_activation": "tanh",
                    "use_lstm": True,
                    "lstm_cell_size": 1024,
                    "max_seq_len": 16,  # LSTM unroll length
                    "vf_share_layers": False,
                    "lstm_use_prev_action": True,
                    "lstm_use_prev_reward": True,
                    "post_fcnet_hiddens": [1024],
                }
            )
            .resources(
                num_gpus=args.num_gpus,
                num_cpus_per_worker=1,
                num_gpus_per_worker=(args.num_gpus / args.num_rollout_workers if args.num_rollout_workers > 0 else 0),
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
    config.debugging(
        logger_config={
            "type": "ray.tune.logger.UnifiedLogger",
            "logdir": os.path.join(output_folder_path, f'results/result_{formatted_time}')
        }
    )

    return config


if __name__ == "__main__":
    # Initialize Ray
    ray.init(ignore_reinit_error=True, num_gpus=args.num_gpus, include_dashboard=False, log_to_driver=True)

    # Get current date and time
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H-%M")
    output_folder_path += f"_{args.algo}_{args.env}"
    # Create output directories
    os.makedirs(output_folder_path, exist_ok=True)
    os.makedirs(os.path.join(output_folder_path, "results"), exist_ok=True)
    os.makedirs(os.path.join(output_folder_path, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(output_folder_path, "results", f"result_{formatted_time}"), exist_ok=True)

    env: Union[PacmanWrapper, ImgObsWrapper] = None
    env_name: str = ""
    callback: Union[MinigridCallback, PacmanCallback]

    if args.env == "pacman":
        from ale_py import ALEInterface

        ale = ALEInterface()
        env_name = "Pacman-CustomPlayground-v0"
        env = PacmanWrapper(ResizeObservation(gym.make("ALE/Pacman-v5", render_mode=args.render_mode, obs_type="rgb"),
                                              shape=(88, 88)))
        callback = PacmanCallback

    if args.env == "minigrid":
        env_name = "MiniGrid-CustomPlayground-v0"
        env = ImgObsWrapper(
            RGBImgPartialObsWrapper(
                TimeLimit(
                    CustomPlaygroundEnv(
                        render_mode=args.render_mode,
                        enable_dowham_reward=args.enable_dowham_reward,
                    ), max_episode_steps=200
                )
            )
        )
        callback = MinigridCallback

    assert isinstance(env, (PacmanWrapper, ImgObsWrapper)), "env must be either PacmanWrapper or ImgObsWrapper"

    register_env(env_name,
                 lambda config: env)
    # Get total CPUs
    total_cpus = os.cpu_count()
    print(f"Total number of available CPUs: {total_cpus}")

    # Get trainer configuration based on algorithm
    config = get_trainer_config(
        env_name,
        args.algo,
        args,
        total_cpus,
        output_folder_path,
        formatted_time,
        callback
    )

    config['observation_space'] = env.observation_space
    config['action_space'] = env.action_space

    # Build the trainer
    trainer = config.build(env=env_name)

    # Restore from checkpoint if needed
    if args.restore and args.checkpoint_path:
        try:
            trainer.restore(args.checkpoint_path)
        except ValueError:
            sys.stdout.write("Checkpoint not found, starting from scratch.\n")

    range_i = range(args.start, args.end + 1)
    if args.use_tqdm:
        range_i = tqdm(range_i)

    # Training loop
    for i in range_i:
        result = trainer.train()

        if i % args.checkpoint_size == 0:
            # Save the model checkpoint
            checkpoint_dir = os.path.join(output_folder_path, "checkpoint")
            checkpoint_path = trainer.save(f'{checkpoint_dir}/checkpoint-{i}')
            print(f"Checkpoint saved at iteration {i} to {checkpoint_path}")

    # Shutdown Ray
    ray.shutdown()
