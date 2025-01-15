import argparse
import datetime
import os
import sys

import gymnasium
import pandas as pd
import ray
from gymnasium import register
from ray.tune import register_env

from callbacks.minigrid.callback import MinigridCallback
from callbacks.pacman.callback import PacmanCallback
from custom_train import get_trainer_config
from environments.minigrid_env import CustomPlaygroundEnv
from environments.minigrid_wrapper import FlattenedPositionWrapper

# Initialize Ray
ray.init(ignore_reinit_error=True, _metrics_export_port=8080)

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
    parser.add_argument('--algo', type=str, help='The algorithm to use (dqn, ppo or impala)', default='dqn')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint Directory', default=None)
    parser.add_argument('--output_folder', type=str, help='Output Folder', default="ray_results")
    parser.add_argument('--env', type=str, help='Environment to run the model on', default="minigrid")
    parser.add_argument('--batch_size', type=int, help='Batch Size', default=32)
    parser.add_argument('--enable_dowham_reward', action='store_true', help='Enable DoWham Intrinsic reward')

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
                 lambda config: FlattenedPositionWrapper(
                     CustomPlaygroundEnv(render_mode=render_mode, enable_dowham_reward_v2=args.enable_dowham_reward)))

    register(
        id="MiniGrid-CustomPlayground-v0",
        entry_point=lambda: FlattenedPositionWrapper(
            CustomPlaygroundEnv(render_mode="human", enable_dowham_reward_v2=True))
    )

    # Get total CPUs
    total_cpus = os.cpu_count()
    print(f"Total number of available CPUs: {total_cpus}")
    # Set up output folder path
    current_dir = os.getcwd()

    # Append the output folder to the current file path
    output_folder_path = os.path.join(os.path.dirname(current_dir), args.output_folder)
    output_folder_path += f"_{args.algo}_{args.env}"
    # Get current date and time
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H-%M")

    if args.env == "pacman":
        callback = PacmanCallback

    if args.env == "minigrid":
        callback = MinigridCallback

    # Get trainer configuration based on algorithm
    config = get_trainer_config(
        "MiniGrid-CustomPlayground-v0",
        args.algo,
        args,
        total_cpus,
        output_folder_path,
        formatted_time,
        callback
    )
    # Get the current working directory
    current_dir = os.getcwd()

    # Define the relative path to the directory where you want to save the model
    relative_path = "checkpoint"

    # Join the current directory with the relative path
    checkpoint_dir = os.path.join(current_dir, relative_path)
    trainer = config.build()
    if not args.checkpoint:
        try:
            trainer.restore(
                f'/Users/berkayeren/ray_results/IMPALA_2025-01-15_21-59-08/DoWhaMV2_batch32[256, 128]42_6db2/checkpoint_000001')
        except ValueError:
            sys.stdout.write("Checkpoint not found, starting from scratch.\n")

    env = gymnasium.make("MiniGrid-CustomPlayground-v0")

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
            action = trainer.compute_single_action(observation=observation, prev_action=action, prev_reward=reward)
            # Take the action in the environment
            observation, reward, done, info, _ = env.step(action)
            if render_mode == 'human':
                env.render()

        # Convert the dictionary to a DataFrame
        date_string = datetime.now().strftime("%Y-%m-%d-%H%M")
        df = pd.DataFrame(list(visited_states.items()))
        df.to_csv(f'trail-algo{args.algo}-episode{episode}-checkpoint{args.checkpoint}-{date_string}.csv', index=False)

    ray.shutdown()
