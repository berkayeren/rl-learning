import random

import numpy as np
from gymnasium.envs.registration import EnvSpec
from gymnasium.spaces import Box, Dict, Discrete
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Door, Key
from minigrid.envs import MultiRoomEnv


class CustomPlaygroundEnv(MultiRoomEnv):
    def __init__(self, intrinsic_reward_scaling=0.05, eta=40, H=1, tau=0.5, size=15):
        self.intrinsic_reward_scaling = intrinsic_reward_scaling
        self.dowham_reward = DoWhaMIntrinsicReward(eta, H, tau)
        super().__init__(minNumRooms=4, maxNumRooms=4, max_steps=200, agent_view_size=size, render_mode='human')

        # Define the observation space to include image, direction, and mission
        self.observation_space = Dict({
            'image': Box(low=0, high=255, shape=(size, size, 3), dtype=np.uint8),
            'direction': Discrete(4),
            'mission': Box(low=0, high=255, shape=(1,), dtype=np.uint8)  # Simplified mission space for demonstration
        })
        self.carrying = Key('yellow')
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
        self.agent_pos = (1, 1)
        self.put_obj(Goal(), 7, 7)

        self.agent_dir = random.randint(0, 3)
        self.mission = "traverse the rooms to get to the goal"

    def step(self, action):
        print('step')
        current_state = self.agent_pos
        obs, reward, done, info, _ = super().step(action)
        next_state = self.agent_pos
        self.dowham_reward.update_state_visits(next_state)
        state_changed = current_state != next_state
        self.dowham_reward.update_usage(action)
        self.dowham_reward.update_effectiveness(action, state_changed)
        intrinsic_reward = self.dowham_reward.calculate_intrinsic_reward(action, current_state, next_state)
        reward += self.intrinsic_reward_scaling * intrinsic_reward

        obs = {
            'image': obs['image'],
            'direction': np.array(self.agent_dir, dtype=np.int64),
            'mission': np.array([ord(c) for c in self.mission[:1]], dtype=np.uint8)  # Simplified mission representation
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

    def update_usage(self, action):
        self.usage_counts[action] = self.usage_counts.get(action, 0) + 1

    def update_effectiveness(self, action, state_changed):
        if state_changed:
            self.effectiveness_counts[action] = self.effectiveness_counts.get(action, 0) + 1

    def calculate_bonus(self, action):
        U = self.usage_counts.get(action, 1)
        E = self.effectiveness_counts.get(action, 0)
        term = (E ** self.H) / (U ** self.H)
        exp_term = self.eta ** (1 - term)
        bonus = (exp_term - 1) / (self.eta - 1)
        return bonus

    def update_state_visits(self, state):
        self.state_visit_counts[state] = self.state_visit_counts.get(state, 0) + 1

    def calculate_intrinsic_reward(self, action, current_state, next_state):
        if current_state != next_state:
            state_count = self.state_visit_counts.get(next_state, 1) ** self.tau
            action_bonus = self.calculate_bonus(action)
            intrinsic_reward = action_bonus / np.sqrt(state_count)
            return intrinsic_reward
        return 0.0

    def reset_episode(self):
        self.state_visit_counts.clear()
