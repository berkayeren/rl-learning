from typing import Dict, Union, Any

import ray
from ray import tune
from ray.rllib.agents.ppo import DEFAULT_CONFIG

# Ensure Ray is shut down and then initialize.
ray.shutdown()
ray.init(ignore_reinit_error=True)

# Define a configuration type for clarity.
ConfigType = Dict[str, Union[str, float, Any]]

# Set up the configuration.
config: ConfigType = DEFAULT_CONFIG.copy()
config['env'] = "BipedalWalker-v3"
config['framework'] = "tf"
config['lr'] = tune.grid_search([5e-05, 5e-08])
config['num_workers'] = 7

# Stopping criteria.
stop: Dict[str, int] = {
    'episode_reward_mean': 300,
}

# Run training using Ray's tune module.
analysis = tune.run(
    'PPO',
    config=config,
    stop=stop,
    checkpoint_at_end=True,
    checkpoint_freq=1000,
    resume=True
)
