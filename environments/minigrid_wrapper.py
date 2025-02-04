import numpy as np
from gymnasium import spaces
from minigrid.wrappers import FullyObsWrapper


class FlattenedPositionWrapper(FullyObsWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            'direction': spaces.Discrete(4),
            'image': spaces.Box(0, 255, env.observation_space["image"].shape, dtype=np.float32),
            'position': spaces.Box(low=0, high=19, shape=(2,), dtype=np.float32),
        })

    def observation(self, observation):
        observation = super().observation(observation)

        return {
            "direction": np.array(observation["direction"], dtype=np.float32),
            "image": np.array(observation["image"], dtype=np.float32),
            "position": np.array(observation["position"], dtype=np.float32),
        }
