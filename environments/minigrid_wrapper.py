import gymnasium as gym
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


class PositionBasedWrapper(gym.Wrapper):
    """
    A wrapper that simplifies observations to only include:
    - Agent position (x, y)
    - Agent direction (0-3)
    - Goal position (x, y)
    """

    def __init__(self, env):
        super().__init__(env)
        print(f"Env Type: {env.env_type}, Max Door: {env.max_door}")
        self.max_door = env.max_door

        if env.max_door == 0:
            self.observation_space = spaces.Dict({
                'agent_pos': spaces.Box(low=0, high=max(env.width, env.height),
                                        shape=(2,), dtype=np.int32),
                'agent_dir': spaces.Discrete(4),
                'goal_pos': spaces.Box(low=0, high=max(env.width, env.height),
                                       shape=(2,), dtype=np.int32),
            })
        else:
            self.observation_space = spaces.Dict({
                'agent_pos': spaces.Box(low=0, high=max(env.width, env.height),
                                        shape=(2,), dtype=np.int32),
                'agent_dir': spaces.Discrete(4),
                'goal_pos': spaces.Box(low=0, high=max(env.width, env.height),
                                       shape=(2,), dtype=np.int32),
                'door_pos': spaces.Box(low=0, high=max(env.width, env.height),
                                       shape=(env.max_door, 2), dtype=np.int32),
                'door_state': spaces.Box(low=0, high=1,
                                         shape=(env.max_door,), dtype=np.int32)
            })

        # Create state visitation counts for exploration bonus
        self.state_visits = {}
        self.exploration_weight = 0.1

    def step(self, action):
        # Execute action in environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._get_position_obs(), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._get_position_obs(), info

    def _get_position_obs(self):
        if self.max_door > 0:
            # Get door positions and states
            door_positions = []
            door_states = []

            # Scan grid for doors
            for i in range(self.env.width):
                for j in range(self.env.height):
                    cell = self.env.grid.get(i, j)
                    if cell is not None and cell.type == 'door':
                        door_positions.append([i, j])
                        # 1 if door is open, 0 if closed
                        door_states.append(1 if cell.is_open else 0)

            # Extract the minimal observation information
            return {
                'agent_pos': np.array(self.env.agent_pos, dtype=np.int32),
                'agent_dir': self.env.agent_dir,
                'goal_pos': np.array(self.env.goal_pos, dtype=np.int32),
                'door_pos': np.array(door_positions, dtype=np.int32),
                'door_state': np.array(door_states, dtype=np.int32)
            }

        return {
            'agent_pos': np.array(self.env.agent_pos, dtype=np.int32),
            'agent_dir': self.env.agent_dir,
            'goal_pos': np.array(self.env.goal_pos, dtype=np.int32)
        }
