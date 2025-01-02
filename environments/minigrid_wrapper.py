import numpy as np
from gymnasium import spaces
from minigrid.wrappers import FullyObsWrapper


class FlattenedPositionWrapper(FullyObsWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Observation space components
        grid_shape = self.observation_space.spaces["image"].shape  # Fully observable grid
        position_size = 2  # (x, y) coordinates
        direction_size = 1  # Scalar direction
        # Flattened observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(np.prod(grid_shape) + position_size + direction_size + env.states.size,),
            dtype=np.float32,
        )

    def observation(self, observation):
        observation = super().observation(observation)

        # Extract components
        grid_flattened = observation["image"].flatten()
        position_flattened = np.array(self.unwrapped.agent_pos, dtype=np.float32)
        direction_flattened = np.array([self.unwrapped.agent_dir], dtype=np.float32)

        # Concatenate all components into a single flat array
        return np.concatenate(
            [grid_flattened, position_flattened, direction_flattened, self.unwrapped.states.flatten()])
