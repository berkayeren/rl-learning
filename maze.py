from ray import tune

# Specify the configuration
config = {
    "env": "classes.environment.GymEnvironment",  # Assuming this is a custom Gym environment you've defined
    "num_workers": 2,
    # ... other PPO-specific configurations
}

if __name__ == "__main__":
    # Train using tune.run() with stopping conditions
    tune.run(
        'PPO',
        config=config,
        stop={
            "timesteps_total": 100000
        }
    )

# rllib evaluate ~/ray_results/maze_env/<checkpoint> --algo DQN\ --env maze_gym_env.Environment\ --steps 100