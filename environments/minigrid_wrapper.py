import numpy as np
from gymnasium import spaces
from minigrid.wrappers import FullyObsWrapper


class FlattenedPositionWrapper(FullyObsWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            'direction': spaces.Discrete(4),
            'image': spaces.Box(0, 255, (np.prod(env.observation_space["image"].shape),), dtype=np.uint8),
            'position': spaces.Box(low=0, high=19, shape=(2,), dtype=np.int64)
        })

    def observation(self, observation):
        observation = super().observation(observation)

        return {
            "image": np.array(observation["image"].flatten(), dtype=np.uint8),
            "position": np.array(observation["position"], dtype=np.int64),
            "direction": np.int64(observation["direction"])
        }
