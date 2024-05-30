from minigrid.wrappers import FlatObsWrapper
from ray.tune import register_env

try:
    import gymnasium as gym

    gymnasium = True
except Exception:
    import gym

    gymnasium = False
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

ray.shutdown()
ray.init(ignore_reinit_error=True, num_gpus=1, _metrics_export_port=8080, include_dashboard=True)


class CustomFlatObsWrapper(FlatObsWrapper):
    def __init__(self, env):
        super().__init__(env)


# Define a custom environment creator function
def env_creator(env_config):
    env = gym.make("MiniGrid-Playground-v0")
    env.reset()
    env = CustomFlatObsWrapper(env)
    return env


# Register the custom environment
register_env("my_minigrid_env", env_creator)

tune_config = (
    PPOConfig()
    .environment("my_minigrid_env")
    .rollouts(
        num_envs_per_worker=20,
        observation_filter="MeanStdFilter",
        num_rollout_workers=0,
    )
    .training()
)

stop = {
    "timesteps_total": 500000,
    "episode_reward_mean": 300.0,
}

# Run training using Ray's tune module.
analysis = tune.run(
    'PPO',
    config=tune_config,
    stop=stop,
    checkpoint_at_end=True,
    checkpoint_freq=10,
    resume="AUTO"
)

# Find the best trial
best_trial = analysis.get_best_trial("mean_reward", "max", "last")
print(f"Best trial config: {best_trial.config}")
print(f"Best trial final reward: {best_trial.last_result['mean_reward']}")
