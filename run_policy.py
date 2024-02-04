from ray.rllib.algorithms.ppo import PPOConfig

from gym_env_trainer import env_creator

# './ray_results/PPO/PPO_my_minigrid_env_0_2020-11-06_10-59-59am2z44t0/checkpoint_500/checkpoint-500'
checkpoint_path = ''

# Restore the trained model from the checkpoint
trained_model = PPOConfig().build(env="my_minigrid_env")
trained_model.restore(checkpoint_path)
# Reuse the env_creator function or directly create the environment
env = env_creator({})
# Reset the environment and get the initial observation
observation = env.reset()
done = False
while not done:
    # Compute an action given the observation
    action = trained_model.compute_single_action(observation)

    # Take a step in the environment
    observation, reward, done, info = env.step(action)

    # Render the environment
    env.render()
