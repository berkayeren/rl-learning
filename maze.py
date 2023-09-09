from ray.rllib.algorithms.dqn import DQNConfig

config = DQNConfig().environment("classes.environment.GymEnvironment").rollouts(num_rollout_workers=2)
