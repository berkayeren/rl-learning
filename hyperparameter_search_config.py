"""
Comprehensive hyperparameter search configuration for PPO with intrinsic motivation.
This configuration is designed to improve training stability and exploration efficiency.
"""

from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.air import CheckpointConfig, FailureConfig
from ray import train


def get_hyperparameter_search_config(base_config, env_type, max_steps, conv_filter=False):
    """
    Returns a comprehensive hyperparameter search space for PPO training.

    Key areas being optimized:
    1. Learning dynamics (lr, batch sizes, epochs)
    2. Exploration (entropy, clipping)
    3. Value function learning (vf_loss_coeff, GAE parameters)
    4. Regularization (grad_clip, KL divergence)
    5. Network architecture
    """

    # Core training hyperparameters
    search_space = {
        **base_config,
        "env_config": {
            "enable_dowham_reward_v2": False,
            "env_type": env_type,
            "max_steps": max_steps,
            "conv_filter": conv_filter,
        },

        # === LEARNING RATE AND OPTIMIZATION ===
        "lr": tune.loguniform(1e-6, 1e-3),  # Critical for stability
        "optimizer": tune.choice([
            {},  # Default Adam
            {"type": "RMSprop", "momentum": 0.0, "epsilon": 0.01},
            {"type": "Adam", "eps": 1e-5},
        ]),

        # === BATCH SIZE AND TRAINING DYNAMICS ===
        "train_batch_size": tune.choice([256, 512, 1024, 2048]),  # Smaller batches for better exploration
        "minibatch_size": tune.choice([64, 128, 256]),  # Should be <= train_batch_size
        "num_epochs": tune.choice([3, 5, 10, 15]),  # How many times to use each batch
        "shuffle_batch_per_epoch": tune.choice([True, False]),

        # === PPO-SPECIFIC PARAMETERS ===
        "clip_param": tune.uniform(0.1, 0.3),  # PPO clipping - critical for stability
        "use_kl_loss": tune.choice([True, False]),
        "kl_coeff": tune.loguniform(0.01, 0.5),  # KL divergence penalty
        "kl_target": tune.uniform(0.005, 0.02),

        # === EXPLORATION AND REGULARIZATION ===
        "entropy_coeff": tune.loguniform(1e-4, 5e-2),  # Encourage exploration
        "vf_loss_coeff": tune.uniform(0.1, 1.0),  # Value function importance
        "grad_clip": tune.uniform(0.5, 10.0),  # Gradient clipping for stability

        # === DISCOUNT AND GAE ===
        "gamma": tune.choice([0.99, 0.995, 0.999]),  # Discount factor
        "lambda": tune.uniform(0.9, 1.0),  # GAE lambda for bias-variance tradeoff
        "use_gae": True,  # Always use GAE for better value estimates

        # === NEURAL NETWORK ARCHITECTURE ===
        "model": tune.choice([
            # Small networks for faster training
            {
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
                "vf_share_layers": False,
                **({"conv_filters": [[16, [3, 3], 2], [32, [3, 3], 2]],
                    "conv_activation": "relu"} if conv_filter else {}),
            },
            # Medium networks - good balance
            {
                "fcnet_hiddens": [128, 128],
                "fcnet_activation": "relu",
                "vf_share_layers": False,
                **({"conv_filters": [[32, [3, 3], 2], [32, [3, 3], 2], [32, [3, 3], 2]],
                    "conv_activation": "elu"} if conv_filter else {}),
            },
            # Larger networks for complex environments
            {
                "fcnet_hiddens": [256, 128],
                "fcnet_activation": "tanh",
                "vf_share_layers": True,  # Share some layers between policy and value
                **({"conv_filters": [[32, [3, 3], 2], [64, [3, 3], 2], [64, [3, 3], 2]],
                    "conv_activation": "elu"} if conv_filter else {}),
            },
            # Alternative architectures
            {
                "fcnet_hiddens": [512, 256],
                "fcnet_activation": "elu",
                "vf_share_layers": False,
                **({"conv_filters": [[32, [3, 3], 2], [32, [3, 3], 2], [32, [3, 3], 2], [32, [3, 3], 2]],
                    "conv_activation": "relu"} if conv_filter else {}),
            },
        ]),
    }

    return search_space


def get_tune_config(num_samples=20):
    """
    Returns the tune configuration for hyperparameter search.
    """
    return tune.TuneConfig(
        metric="env_runners/episode_len_mean",  # Minimize steps to reach goal
        mode="min",
        num_samples=num_samples,
        reuse_actors=True,
        search_alg=OptunaSearch(),  # More efficient than grid search
        max_concurrent_trials=4,  # Adjust based on your resources
    )


def get_run_config(timesteps_total=1_000_000):
    """
    Returns the run configuration for the experiment.
    """
    return train.RunConfig(
        stop={"timesteps_total": timesteps_total},
        failure_config=FailureConfig(max_failures=3),
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_frequency=20,
            checkpoint_at_end=True,
            checkpoint_score_attribute="env_runners/episode_len_mean",
            checkpoint_score_order="min"
        )
    )


# === SPECIFIC HYPERPARAMETER COMBINATIONS FOR DIFFERENT SCENARIOS ===

def get_fast_exploration_config(base_config, env_type, max_steps):
    """
    Configuration optimized for fast exploration in sparse reward environments.
    """
    return {
        **base_config,
        "env_config": {
            "enable_dowham_reward_v2": True,  # Use intrinsic motivation
            "env_type": env_type,
            "max_steps": max_steps,
        },
        "lr": tune.loguniform(1e-5, 1e-3),
        "entropy_coeff": tune.loguniform(1e-3, 1e-1),  # Higher entropy for exploration
        "train_batch_size": tune.choice([512, 1024]),  # Smaller batches
        "clip_param": tune.uniform(0.15, 0.25),  # Moderate clipping
        "gamma": 0.99,  # Standard discount
        "model": {
            "fcnet_hiddens": tune.choice([[128, 128], [256, 128]]),
            "fcnet_activation": tune.choice(["relu", "elu"]),
            "vf_share_layers": False,
        }
    }


def get_stable_learning_config(base_config, env_type, max_steps):
    """
    Configuration optimized for stable learning with lower variance.
    """
    return {
        **base_config,
        "env_config": {
            "enable_dowham_reward_v2": False,
            "env_type": env_type,
            "max_steps": max_steps,
        },
        "lr": tune.loguniform(1e-6, 5e-4),  # Lower learning rates
        "train_batch_size": tune.choice([1024, 2048, 4096]),  # Larger batches
        "num_epochs": tune.choice([10, 15, 20]),  # More epochs
        "clip_param": tune.uniform(0.1, 0.2),  # Conservative clipping
        "entropy_coeff": tune.loguniform(1e-4, 1e-2),  # Lower entropy
        "vf_loss_coeff": tune.uniform(0.5, 1.0),  # Higher value function weight
        "grad_clip": tune.uniform(0.5, 5.0),  # Tighter gradient clipping
        "use_kl_loss": True,
        "kl_coeff": tune.uniform(0.1, 0.5),
    }


# === SAMPLE USAGE ===
if __name__ == "__main__":
    # Example of how to use these configurations
    print("Hyperparameter search configurations available:")
    print("1. get_hyperparameter_search_config() - Comprehensive search")
    print("2. get_fast_exploration_config() - Optimized for exploration")
    print("3. get_stable_learning_config() - Optimized for stability")
