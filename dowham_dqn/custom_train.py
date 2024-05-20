import ray
from gymnasium.envs.registration import register
from ray.rllib.algorithms import DQN
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from dowham_dqn.custom_dqn_model import CustomDQNModel
from dowham_dqn.custom_playground_env import CustomPlaygroundEnv

# Initialize Ray
ray.init(ignore_reinit_error=True)

# class CustomFlatObsWrapper(FlatObsWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#
#
# def env_creator(env_config=None):
#     config = {
#         "agent_start_pos": (1, 1),
#         "agent_start_dir": 0,
#         "goal_pos": (15, 15),
#         "minNumRooms": 2,
#         "maxNumRooms": 5,
#         "enable_dowham": True,
#         "max_episode_steps": 1000,
#         **env_config
#     }
#     env = CustomPlaygroundEnv()
#     env.reset()
#     env = CustomFlatObsWrapper(env)
#     return env


# Register the custom environment
# register_env("my_minigrid_env", env_creator)

# Register the custom environment
register_env("MiniGrid-CustomPlayground-v0", lambda config: CustomPlaygroundEnv())

# Configure DQN with the custom environment and model
config = (
    DQNConfig()
    .environment(env="MiniGrid-CustomPlayground-v0")
    .rollouts(num_rollout_workers=1)  # Adjust the number of workers as needed
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
        lr=1e-5,
        optimizer={"type": "RMSProp"},
        model={
            "custom_model": "custom_dqn_model",
        },
        gamma=0.99,
        train_batch_size=32,
        # entropy_coeff=0.001,
        num_atoms=1,
        v_min=-10.0,
        v_max=10.0,
        noisy=False,
        dueling=True,
        double_q=True,
        n_step=3,
        target_network_update_freq=500,
    )
    .resources(
        num_gpus=0,
        num_cpus_per_worker=1
    )
)

# Register the custom model
ModelCatalog.register_custom_model("custom_dqn_model", CustomDQNModel)

# Instantiate the DQN trainer
dqn_trainer = DQN(config=config)

if __name__ == "__main__":
    # Register the custom environment
    register(
        id='MiniGrid-CustomPlayground-v0',
        entry_point=CustomPlaygroundEnv,
    )

    # Training loop
    for i in range(1000):  # Number of training iterations
        result = dqn_trainer.train()
        print(f"Iteration {i} - Reward: {result['episode_reward_mean']}")

    # Save the trained model
    checkpoint = dqn_trainer.save()
    print(f"Checkpoint saved at {checkpoint}")

    # Shutdown Ray
    ray.shutdown()
