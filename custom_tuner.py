import argparse
import datetime
import os
from functools import partial

import gymnasium as gym
import ray
from gymnasium.wrappers import TimeLimit
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from ray import train
from ray import tune
from ray.tune import register_env, PlacementGroupFactory
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from callbacks.minigrid.callback import MinigridCallback
from callbacks.pacman.callback import PacmanCallback
from custom_train import get_trainer_config
from environments.minigrid_env import CustomPlaygroundEnv
from environments.pacman_env import PacmanWrapper

os.environ['PYTHONWARNINGS'] = "ignore::DeprecationWarning"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ['PYTHONUNBUFFERED'] = "1"


# Register the custom environment
def get_environment(env_name: str, args):
    if env_name == "pacman":
        env = PacmanWrapper(
            gym.wrappers.ResizeObservation(
                gym.make("ALE/Pacman-v5", render_mode=args.render_mode, obs_type="rgb"),
                shape=(88, 88)
            )
        )
        callback = PacmanCallback
    elif env_name == "minigrid":
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
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return env, callback


# Hyperparameter search space
def get_search_space(algo_name):
    if algo_name.lower() == 'dqn':
        return {
            "lr": tune.loguniform(1e-5, 1e-2),
            "train_batch_size": tune.choice([32, 64, 128]),
            "rollout_fragment_length": tune.choice([50, 100, 200]),
            "n_step": tune.choice([1, 3, 5]),
            "exploration_config.final_epsilon": tune.uniform(0.01, 0.1),
            "exploration_config.epsilon_timesteps": tune.randint(5000, 20000),
        }
    elif algo_name.lower() == 'ppo':
        return {
            "lr": tune.loguniform(1e-5, 1e-2),
            "train_batch_size": tune.choice([4000, 8000]),
            "sgd_minibatch_size": tune.choice([64, 128]),
            "num_sgd_iter": tune.choice([10, 20, 30]),
            "clip_param": tune.uniform(0.1, 0.3),
            "entropy_coeff": tune.uniform(0.01, 0.1),
        }
    elif algo_name.lower() == 'impala':
        return {
            "lr": tune.loguniform(1e-5, 1e-2),
            "train_batch_size": tune.choice([2000, 4000]),
            "entropy_coeff": tune.uniform(0.001, 0.01),
            "vf_loss_coeff": tune.uniform(10, 50),
            "grad_clip": tune.choice([40, 42, 50]),
            "rollout_fragment_length": tune.choice([50, 100, 200]),
        }
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


# Trainable function
def trainable(config, env_name, callback_cls, algo_name, args):
    total_cpus = os.cpu_count()
    current_dir = os.getcwd()
    # Get current date and time
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H-%M")

    output_folder_path = os.path.join(os.path.dirname(current_dir), args.output_folder)

    # Get trainer configuration based on algorithm
    algo_config = get_trainer_config(
        env_name,
        algo_name,
        args,
        total_cpus,
        output_folder_path,
        formatted_time,
        callback_cls
    )

    trainer = algo_config.build()
    for _ in range(args.max_iters):
        result = trainer.train()
        # Extract the custom metrics if they exist
        custom_metrics = result.get("custom_metrics", {})
        percentage_visited_mean = custom_metrics.get("percentage_visited_mean", float('nan'))
        intrinsic_reward_mean = custom_metrics.get("intrinsic_reward_mean", float('nan'))

        train.report({
            "episode_reward_mean": result["episode_reward_mean"],
            "timesteps_total": result["timesteps_total"],
            "percentage_visited_mean": percentage_visited_mean,
            "intrinsic_reward_mean": intrinsic_reward_mean
        })
    trainer.stop()


# Main logic
if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Hyperparameter tuning script")
    parser.add_argument('--algo', type=str, help='The algorithm to use (dqn, ppo, or impala)', default='dqn')
    parser.add_argument('--env', type=str, help='The environment to use (minigrid or pacman)', default='minigrid')
    parser.add_argument('--num_samples', type=int, help='Number of samples for hyperparameter search', default=10)
    parser.add_argument('--max_iters', type=int, help='Maximum iterations for training', default=100)
    parser.add_argument('--output_folder', type=str, help='Output folder', default="tune_results")
    parser.add_argument('--enable_dowham_reward', action='store_true', help='Enable DoWham Intrinsic reward')
    parser.add_argument('--num_rollout_workers', type=int, help='The number of rollout workers', default=1)
    parser.add_argument('--num_envs_per_worker', type=int, help='The number of environments per worker', default=1)
    parser.add_argument('--num_gpus', type=int, help='The number of GPUs', default=0)
    parser.add_argument('--render_mode', type=str, help='The render mode parameter', default=None)
    args = parser.parse_args()

    current_dir = os.getcwd()
    output_folder_path = os.path.join(os.path.dirname(current_dir), args.output_folder)
    output_folder_path += f"_{args.algo}_{args.env}"

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Get environment and callback
    env, callback_cls = get_environment(args.env, args)
    register_env(args.env, lambda config: env)

    # Search space
    search_space = get_search_space(args.algo)

    # Scheduler and search algorithm
    scheduler = ASHAScheduler(
        metric="percentage_visited_mean",
        mode="max",
        grace_period=10,
        max_t=args.max_iters
    )
    search_algo = OptunaSearch(
        metric="percentage_visited_mean",
        mode="max"
    )

    total_cpus = os.cpu_count()
    # Tune run
    analysis = tune.run(
        partial(trainable, env_name=args.env, callback_cls=callback_cls, algo_name=args.algo, args=args),
        config=search_space,
        num_samples=args.num_samples,
        scheduler=scheduler,
        search_alg=search_algo,
        local_dir=output_folder_path,
        name=f"Tune_{args.algo}_{args.env}",
        resources_per_trial=PlacementGroupFactory([{"CPU": 1}] + [{"CPU": 1}] * (
                total_cpus - 1)),
    )

    # After tune.run finishes:
    best_config = analysis.get_best_config(metric="percentage_visited_mean", mode="max")
    print("Best config:", best_config)
