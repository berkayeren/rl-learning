import numpy as np
from gymnasium import spaces
from minigrid.wrappers import FullyObsWrapper
from numpy.linalg import norm


class FlattenedPositionWrapper(FullyObsWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            'direction': spaces.Discrete(4),
            'image': spaces.Box(0, 255, (np.prod(env.observation_space["image"].shape),), dtype=np.uint8),
            'position': spaces.Box(low=0, high=19, shape=(2,), dtype=np.int64),
            'goal_distance': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })

    def observation(self, observation):
        observation = super().observation(observation)
        agent_position = np.array(observation["position"], dtype=np.float32)
        goal_distance = norm(self.env.goal_pos - agent_position)  # Euclidean distance

        return {
            "image": np.array(observation["image"].flatten(), dtype=np.uint8),
            "position": np.array(observation["position"], dtype=np.int64),
            "direction": np.int64(observation["direction"]),
            "goal_distance": np.array([goal_distance], dtype=np.float32)

        }
