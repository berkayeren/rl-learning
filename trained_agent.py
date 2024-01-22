from typing import Dict, Any, Union

import gym
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG

# Define a configuration type for clarity.
ConfigType = Dict[str, Union[str, float, Any]]

# Set up the configuration.
config = DEFAULT_CONFIG.copy()
config['env'] = "BipedalWalker-v3"
config['framework'] = "tf"
config['lr'] = tune.grid_search([5e-05, 5e-08])
config["lr"] = 5e-08

# Create a new trainer and restore from checkpoint.
new_trainer: PPOTrainer = PPOTrainer(config=config)

# Restore trainer from a point.
# new_trainer.restore(
#     "C:\\Users\\BerkayEren\\ray_results\\PPO\\PPO_BipedalWalker-v3_2bf94_00000_0_lr=0.0001_2023-10-11_23-02-58\\checkpoint_000125"
# )

# Set up the environment.
env = gym.make("MiniGrid-Playground-v0")
env.reset()
observation = env.reset()
action: gym.spaces.Box = new_trainer.compute_single_action(observation)
observation, reward, done, info = env.step(action)

# Run the episode.
while True:
    env.render()
    print(observation)
    action = new_trainer.compute_single_action(observation, prev_action=action, prev_reward=reward)
    observation, reward, done, info = env.step(action)
    if done:
        # If the episode is over.
        break
env.close()
