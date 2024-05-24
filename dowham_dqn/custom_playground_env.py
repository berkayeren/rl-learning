import random

import numpy as np
from gymnasium.envs.registration import EnvSpec
from gymnasium.spaces import Box, Dict, Discrete
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Door, Key
from minigrid.envs import MultiRoomEnv


def hash_dict(d):
    hashable_items = []
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            value = value.tobytes()
        hashable_items.append((key, value))
    return abs(hash(tuple(sorted(hashable_items))))


class CustomPlaygroundEnv(MultiRoomEnv):
    def __init__(self, intrinsic_reward_scaling=0.05, eta=40, H=1, tau=0.5, size=15):
        self.intrinsic_reward_scaling = intrinsic_reward_scaling
        self.dowham_reward = DoWhaMIntrinsicReward(eta, H, tau)
        super().__init__(minNumRooms=4, maxNumRooms=4, max_steps=200, agent_view_size=size)

        # Define the observation space to include image, direction, and mission
        self.observation_space = Dict({
            'image': Box(low=0, high=255, shape=(size, size, 3), dtype=np.uint8),
            'direction': Discrete(4),
            'mission': Box(low=0, high=255, shape=(1,), dtype=np.uint8)  # Simplified mission space for demonstration
        })
        # self.carrying = Key('yellow')
        self.spec = EnvSpec("CustomPlaygroundEnv-v0", max_episode_steps=200)

    @staticmethod
    def _gen_mission():
        return "traverse the rooms to get to the goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        # Place a vertical wall to divide the grid into two halves
        self.grid.vert_wall(width // 2, 0, height)

        # Place a horizontal wall to divide the grid into two halves
        self.grid.horz_wall(0, height // 2, width)
        self.put_obj(Door('yellow'), width // 2, height // 4)  # Door in the upper part of the vertical wall
        self.put_obj(Door('yellow'), width // 2, 3 * height // 4)  # Door in the lower part of the vertical wall
        self.put_obj(Door('yellow'), width // 4, height // 2)  # Door in the left part of the horizontal wall
        self.put_obj(Door('yellow'), 3 * width // 4, height // 2)  # Door in the right part of the horizontal wall
        self.put_obj(Key('yellow'), 3, 3)
        self.agent_pos = (1, 1)
        self.put_obj(Goal(), 14, 7)

        self.agent_dir = random.randint(0, 3)
        self.mission = "traverse the rooms to get to the goal"

    def step(self, action):
        current_state = self.agent_pos
        current_obs = self.gen_obs()
        current_obs = hash_dict(current_obs)
        obs, reward, done, info, _ = super().step(action)
        next_state = self.agent_pos
        next_obs = hash_dict(obs)
        self.dowham_reward.update_state_visits(current_obs, next_obs)
        state_changed = current_state != next_state
        self.dowham_reward.update_usage(current_obs, action)
        self.dowham_reward.update_effectiveness(current_obs, action, next_obs, state_changed)
        intrinsic_reward = self.dowham_reward.calculate_intrinsic_reward(current_obs, action, next_obs)
        reward += self.intrinsic_reward_scaling * intrinsic_reward
        obs = {
            'image': obs['image'],
            'direction': np.array(self.agent_dir, dtype=np.int64),
            'mission': np.array([ord(c) for c in self.mission[:1]], dtype=np.uint8)
        }
        return obs, reward, done, info, {}

    def reset(self, **kwargs):
        self.dowham_reward.reset_episode()
        obs = super().reset(**kwargs)
        obs = {
            'image': obs[0]['image'],
            'direction': np.array(self.agent_dir, dtype=np.int64),
            'mission': np.array([ord(c) for c in self.mission[:1]], dtype=np.uint8)  # Simplified mission representation
        }

        return obs, {}


class DoWhaMIntrinsicReward:
    def __init__(self, eta, H, tau):
        self.eta = eta
        self.H = H
        self.tau = tau
        self.usage_counts = {}
        self.effectiveness_counts = {}
        self.state_visit_counts = {}

    def update_usage(self, obs, action):
        if obs not in self.usage_counts:
            self.usage_counts[obs] = {}
        self.usage_counts[obs][action] = self.usage_counts[obs].get(action, 0) + 1

    def update_effectiveness(self, obs, action, next_obs, state_changed):
        if obs not in self.effectiveness_counts:
            self.effectiveness_counts[obs] = {}

        if action not in self.effectiveness_counts[obs]:
            self.effectiveness_counts[obs][action] = 0

        if state_changed and obs != next_obs:
            self.effectiveness_counts[obs][action] += 1

    def calculate_bonus(self, obs, action):
        if obs not in self.usage_counts or obs not in self.effectiveness_counts:
            return 0

        U = self.usage_counts[obs].get(action, 1)
        E = self.effectiveness_counts[obs].get(action, 0)
        term = (E ** self.H) / (U ** self.H)
        exp_term = self.eta ** (1 - term)
        bonus = (exp_term - 1) / (self.eta - 1)
        return bonus

    def update_state_visits(self, current_obs, next_obs):
        if current_obs not in self.state_visit_counts:
            self.state_visit_counts[current_obs] = 0

        if next_obs not in self.state_visit_counts:
            self.state_visit_counts[next_obs] = 0

        self.state_visit_counts[current_obs] += 1

    def calculate_intrinsic_reward(self, obs, action, next_obs):
        reward = 0.0
        if obs != next_obs:
            if action == 3:
                reward += 5.0  # Higher reward for taking the key
            if action == 5:
                reward += 10.0  # Higher reward for opening the door

            state_count = self.state_visit_counts[obs] ** self.tau
            action_bonus = self.calculate_bonus(obs, action)
            intrinsic_reward = action_bonus / np.sqrt(state_count)
            return intrinsic_reward + reward
        return 0.0

    def reset_episode(self):
        self.usage_counts.clear()
        self.effectiveness_counts.clear()
        self.state_visit_counts.clear()
