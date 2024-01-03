import gym
import numpy as np
import tensorflow as tf
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.utils.framework import try_import_tf
from ray.tune.registry import register_env

tf1, tf, tfv = try_import_tf()


class CustomTFModel(TFModelV2):
    """Custom model for MiniGrid observations."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomTFModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.base_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(action_space.n, activation=None)
        ])

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]["image"]
        obs = tf.reshape(obs, [-1, np.prod(obs.shape[1:])])  # Flatten the image
        return self.base_model(obs), state

    def value_function(self):
        return tf.reshape(self.base_model.layers[-1].output, [-1])


ModelCatalog.register_custom_model("custom_tf_model", CustomTFModel)


class CustomTFPolicy(TFPolicy):
    def __init__(self, observation_space, action_space, config):
        super(CustomTFPolicy, self).__init__(observation_space, action_space, config)
        self.model = CustomTFModel(observation_space, action_space, action_space.n, config, "custom_tf_model")

    def compute_actions(self, obs_batch, **kwargs):
        model_out, _ = self.model.forward({"obs": obs_batch}, [], None)
        return np.argmax(model_out.numpy(), axis=1), [], {}


if __name__ == "__main__":
    tf1, tf, tfv = try_import_tf()


    def env_creator(env_config):
        return gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")


    register_env("my_minigrid", env_creator)

    ModelCatalog.register_custom_model("my_model", CustomTFPolicy)
    trainer_config = {
        "env": "my_minigrid",
        "model": {
            "custom_model": "my_model",
        },
        "num_gpus": 1,
        "framework": "tf",
    }

    trainer = PPOTrainer(config=trainer_config)

    analysis = tune.run(
        "PPO",
        stop={"training_iteration": 100},
        config=trainer_config,
        verbose=1
    )

    # Accessing the best model or analyzing results
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_checkpoint = analysis.get_best_checkpoint(best_trial)
