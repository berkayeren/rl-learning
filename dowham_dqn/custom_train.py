import argparse
import os

import ray
from minigrid.wrappers import ImgObsWrapper
from ray.rllib.algorithms import DQN
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from dowham_dqn.custom_dqn_model import MinigridPolicyNet
from dowham_dqn.custom_playground_env import CustomPlaygroundEnv

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Register the custom model
ModelCatalog.register_custom_model("MinigridPolicyNet", MinigridPolicyNet)

if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description="Get the render mode parameter")

    # Add the arguments
    parser.add_argument('--render_mode', type=str, help='The render mode parameter', default=None)
    parser.add_argument('--num_rollout_workers', type=int, help='The number of rollout workers', default=1)
    parser.add_argument('--num_envs_per_worker', type=int, help='The number of environments per worker', default=1)
    parser.add_argument('--num_gpus', type=int, help='The number of environments per worker', default=0)

    # Parse the arguments
    args = parser.parse_args()

    # Get the parameters
    render_mode = args.render_mode
    num_rollout_workers = args.num_rollout_workers
    num_envs_per_worker = args.num_envs_per_worker
    num_gpus = args.num_gpus

    # Register the custom environment
    register_env("MiniGrid-CustomPlayground-v0",
                 lambda config: ImgObsWrapper(CustomPlaygroundEnv(render_mode=render_mode)))

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

    # Training loop
    for i in range(1000):  # Number of training iterations
        print(f"Iteration {i}")
        result = dqn_trainer.train()
        print(f"Iteration {i} - Reward: {result['episode_reward_mean']}")
        # Save the trained model
        checkpoint = dqn_trainer.save(f'{checkpoint_dir}/checkpoint-{i}')

    # Shutdown Ray
    ray.shutdown()
