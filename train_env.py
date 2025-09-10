"""Example of defining a custom gymnasium Env to be learned by an RLlib Algorithm.

This example:
    - demonstrates how to write your own (single-agent) gymnasium Env class, define its
    physics and mechanics, the reward function used, the allowed actions (action space),
    and the type of observations (observation space), etc..
    - shows how to configure and setup this environment class within an RLlib
    Algorithm config.
    - runs the experiment with the configured algo, trying to solve the environment.

To see more details on which env we are building for this example, take a look at the
`SimpleCorridor` class defined below.


How to run this script
----------------------
`python [script file name].py`

Use the `--corridor-length` option to set a custom length for the corridor. Note that
for extremely long corridors, the algorithm should take longer to learn.

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`


Results to expect
-----------------
You should see results similar to the following in your console output:

+--------------------------------+------------+-----------------+--------+
| Trial name                     | status     | loc             |   iter |
|--------------------------------+------------+-----------------+--------+
| PPO_SimpleCorridor_78714_00000 | TERMINATED | 127.0.0.1:85794 |      7 |
+--------------------------------+------------+-----------------+--------+

+------------------+-------+----------+--------------------+
|   total time (s) |    ts |   reward |   episode_len_mean |
|------------------+-------+----------+--------------------|
|          18.3034 | 28000 | 0.908918 |            12.9676 |
+------------------+-------+----------+--------------------+
"""
# These tags allow extracting portions of this script on Anyscale.
# ws-template-imports-start
import gymnasium as gym

from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper, PositionBonus
# ws-template-imports-end

from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env

from environments.empty import EmptyEnv
from trainer import CustomEnv, CustomCallback

parser = add_rllib_example_script_args(
    default_iters=3000,
    default_reward=3000,
    default_timesteps=10_000_000,
)

# ws-template-code-end

if __name__ == "__main__":
    args = parser.parse_args()

    register_env(
        "CustomEnv",
        lambda config: ImgObsWrapper(RGBImgPartialObsWrapper(CustomEnv(**config), tile_size=12))
    )
    gym.register("CustomEnv", entry_point="trainer:CustomEnv")
    env = gym.make(
        id="CustomEnv",
        max_episode_steps=1444,
        disable_env_checker=False,
        max_steps=1444,
        size=19,
        env_type=CustomEnv.Environments.multi_room,
        enable_dowham_reward_v2=True,
        highlight=True,
    )
    env = ImgObsWrapper(RGBImgPartialObsWrapper(env, tile_size=12))
    obs = env.reset()
    env.render()
    print(f"Env observation space: {env.observation_space}")
    print(f"Env action space: {env.action_space}")

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment(
            "CustomEnv",  # or provide the registered string: "corridor-env"
            env_config={"max_steps": 1444,
                        "size": 19, "tile_size": 12, "env_type": CustomEnv.Environments.multi_room,
                        "enable_dowham_reward_v2": True},
            disable_env_checking=True,
            is_atari=False,
            observation_space=env.observation_space,
            action_space=env.action_space,
        ).experimental(
            _disable_preprocessor_api=True, )
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=8,
            batch_mode="truncate_episodes",
            rollout_fragment_length=64,
        ).learners(
            num_learners=1,
            num_cpus_per_learner=1,
        )
        .training(
            use_critic=True,
            use_gae=True,
            use_kl_loss=True,
            kl_coeff=0.2,
            kl_target=0.01,
            vf_loss_coeff=0.5,
            entropy_coeff=0.006,
            train_batch_size_per_learner=16384,
            minibatch_size=2048,
            entropy_coeff_schedule=[
                [0, 0.006],
                [5e5, 0.002],
                [1.2e6, 0.0],
            ],
            clip_param=0.3,
            vf_clip_param=10.0,
            lr_schedule=None,
            lr=2.5e-4,
            lambda_=0.95,
            gamma=0.99,
            num_epochs=6,
            model={'fcnet_hiddens': [512, 512],
                   'fcnet_activation': 'tanh',
                   'fcnet_weights_initializer': None,
                   'fcnet_weights_initializer_config': None,
                   'fcnet_bias_initializer': None,
                   'fcnet_bias_initializer_config': None,
                   # "conv_filters": [
                   #     [16, [3, 3], 5],
                   #     [32, [4, 4], 1],
                   # ],
                   'conv_activation': 'relu',
                   'conv_kernel_initializer': None,
                   'conv_kernel_initializer_config': None,
                   'conv_bias_initializer': None,
                   'conv_bias_initializer_config': None,
                   'conv_transpose_kernel_initializer': None,
                   'conv_transpose_kernel_initializer_config': None,
                   'conv_transpose_bias_initializer': None,
                   'conv_transpose_bias_initializer_config': None,
                   'post_fcnet_hiddens': [512],
                   'post_fcnet_activation': 'relu',
                   'post_fcnet_weights_initializer': None,
                   'post_fcnet_weights_initializer_config': None,
                   'post_fcnet_bias_initializer': None,
                   'post_fcnet_bias_initializer_config': None,
                   'free_log_std': False,
                   'log_std_clip_param': 20.0,
                   'no_final_linear': False,
                   'vf_share_layers': False,
                   'use_lstm': True,
                   'max_seq_len': 64,
                   'lstm_cell_size': 256,
                   'lstm_use_prev_action': True,
                   'lstm_use_prev_reward': True,
                   'lstm_weights_initializer': None,
                   'lstm_weights_initializer_config': None,
                   'lstm_bias_initializer': None,
                   'lstm_bias_initializer_config': None,
                   '_time_major': False,
                   'use_attention': False,
                   'attention_num_transformer_units': 1,
                   'attention_dim': 64,
                   'attention_num_heads': 1,
                   'attention_head_dim': 32,
                   'attention_memory_inference': 50,
                   'attention_memory_training': 50,
                   'attention_position_wise_mlp_dim': 32,
                   'attention_init_gru_gate_bias': 2.0,
                   'attention_use_n_prev_actions': 0,
                   'attention_use_n_prev_rewards': 0,
                   'framestack': False,
                   'dim': 88,
                   'grayscale': False,
                   'zero_mean': True,
                   'custom_model': None,
                   'custom_model_config': {},
                   'custom_action_dist': None,
                   'custom_preprocessor': None,
                   'encoder_latent_dim': None,
                   'always_check_shapes': False,
                   '_disable_preprocessor_api': False,
                   '_disable_action_flattening': False}

        ).callbacks(CustomCallback)

    )

    run_rllib_example_script_experiment(base_config, args)
