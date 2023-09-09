from ray.rllib.algorithms.dqn import DQNConfig

config = DQNConfig().environment("classes.environment.GymEnvironment").rollouts(num_rollout_workers=2)

# rllib train file maze.py --stop '{\"timesteps_total\": 10000}'
