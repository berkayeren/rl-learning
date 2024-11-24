import collections
import hashlib
import random
from collections import defaultdict

import numpy as np
from gymnasium.envs.registration import EnvSpec
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Door, Key
from minigrid.envs import FourRoomsEnv


def hash_dict(d):
    hashable_items = []
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            value = value.tobytes()
        hashable_items.append((key, value))
    return abs(hash(tuple(sorted(hashable_items))))


import torch
import torch.nn as nn


class MiniGridNet(nn.Module):
    def __init__(self):
        super(MiniGridNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7 + 1, 128)
        self.fc2 = nn.Linear(128, 7)  # 7 possible actions

    def forward(self, x, direction):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, direction), dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class CustomPlaygroundEnv(FourRoomsEnv):
    def __str__(self):
        if self.agent_pos is None:
            self.reset()

        return super().__str__()

    def __init__(self, intrinsic_reward_scaling=0.05, eta=40, H=1, tau=0.5, size=7, render_mode=None, **kwargs):
        self.agent_pos = (1, 1)
        self.goal_pos = (16, 16)
        self.agent_dir = 0
        self.success_rate: float = 0.0
        self.done = False
        self.intrinsic_reward_scaling = intrinsic_reward_scaling
        self.enable_dowham_reward = kwargs.pop('enable_dowham_reward', None)
        self.enable_count_based = kwargs.pop('enable_count_based', None)
        self.action_count = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
        }

        if self.enable_dowham_reward:
            print("Enabling DoWhaM intrinsic reward")
            self.dowham_reward = DoWhaMIntrinsicReward(eta, H, tau)
            self.intrinsic_reward = 0.0

        if self.enable_count_based:
            print("Enabling count-based exploration")
            self.count_exploration = CountExploration(self, gamma=0.99, epsilon=0.1, alpha=0.1)
            self.count_bonus = 0.0

        self.episode_history = []

        self.minNumRooms = kwargs.pop('minNumRooms', 4)
        self.maxNumRooms = kwargs.pop('maxNumRooms', 4)
        self.maxRoomSize = kwargs.pop('maxRoomSize', 10)
        self.max_possible_rooms = kwargs.pop('max_possible_rooms', 6)

        super().__init__(
            max_steps=200, agent_view_size=size, render_mode=render_mode,
            agent_pos=self.agent_pos, goal_pos=self.goal_pos,
            **kwargs
        )

        self.spec = EnvSpec("CustomPlaygroundEnv-v0", max_episode_steps=200)
        self.success_history = collections.deque(maxlen=1024)

    @staticmethod
    def _gen_mission():
        return "traverse the rooms to get to the goal"

    def __gen_grid(self, width, height):
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

    def observations_changed(self, current_obs, next_obs):
        # Compare images
        images_equal = np.array_equal(current_obs['image'], next_obs['image'])
        # Compare directions
        directions_equal = current_obs['direction'] == next_obs['direction']
        # You can include mission if it changes over time
        # For MiniGrid, the mission usually remains the same
        return not (images_equal and directions_equal)

    def hash(self, size=32):
        """Compute a hash that uniquely identifies the current state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()
        grid, vis_mask = self.gen_obs_grid(agent_view_size=self.size)

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        to_encode = [image.tolist(), self.agent_pos, self.agent_dir]
        for item in to_encode:
            sample_hash.update(str(item).encode("utf8"))

        return sample_hash.hexdigest()[:size]

    def step(self, action):
        self.action_count[action] += 1
        current_state = self.hash()
        current_obs = self.gen_obs()

        obs, reward, done, info, _ = super().step(action)

        next_state = self.hash()
        next_obs = self.gen_obs()

        if self.enable_dowham_reward:
            self.dowham_reward.update_state_visits(current_state, next_state)
            state_changed = self.observations_changed(current_obs, next_obs)
            self.dowham_reward.update_usage(current_state, action)
            self.dowham_reward.update_effectiveness(current_state, action, next_state, state_changed)
            intrinsic_reward = self.dowham_reward.calculate_intrinsic_reward(current_state, action, next_state,
                                                                             state_changed)
            self.intrinsic_reward = self.intrinsic_reward_scaling * intrinsic_reward
            reward += self.intrinsic_reward
        if self.enable_count_based:
            bonus = self.count_exploration.update(current_obs, action, reward, next_obs)
            self.count_bonus = bonus
            reward += bonus

        obs = {
            'image': obs['image'],
            'direction': np.array(self.agent_dir, dtype=np.int64),
            'mission': np.array([ord(c) for c in self.mission[:1]], dtype=np.uint8)
        }

        self.done = done

        if done:
            reward += max(10, 100 / np.sqrt(sum(self.action_count.values())))

        return obs, reward, done, info, {}

    def reset(self, **kwargs):
        self.episode_history = []
        if self.enable_dowham_reward:
            self.dowham_reward.reset_episode()

        if self.enable_count_based:
            self.count_exploration.reset()

        self.action_count = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
        }
        self.success_history.append(self.done)
        self.success_rate = sum(self.success_history) / len(self.success_history)

        obs = super().reset(**kwargs)
        self.done = False
        obs = {
            'image': obs[0]['image'],
            'direction': np.array(self.agent_dir, dtype=np.int64),
            'mission': np.array([ord(c) for c in self.mission[:1]], dtype=np.uint8)  # Simplified mission representation
        }

        return obs, {}


class Count:
    def __init__(self):
        self.counts = defaultdict(int)

    def increment(self, state, action):
        self.counts[(state, action)] += 1

    def get_count(self, state, action):
        return self.counts[(state, action)]


class CountExploration:
    def __init__(self, env, gamma=0.99, epsilon=0.1, alpha=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.count = Count()
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))

    def update(self, state, action, reward, next_state):
        self.count.increment(state, action)
        count = self.count.get_count(state, action)
        bonus = 1.0 / np.sqrt(count)
        self.q_table[state][action] += self.alpha * (
                reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action] + bonus)
        return bonus

    def reset(self):
        self.count = Count()
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))


class DoWhaMIntrinsicReward:
    def __init__(self, eta, H, tau):
        self.eta = eta
        self.H = H
        self.tau = tau
        self.usage_counts = {}
        self.effectiveness_counts = {}
        self.state_visit_counts = {}
        self.recent_transitions = collections.deque(maxlen=64)  # Track recent state transitions

    def update_usage(self, obs, action):
        if obs not in self.usage_counts:
            self.usage_counts[obs] = {}
        self.usage_counts[obs][action] = self.usage_counts[obs].get(action, 0) + 1

    def update_effectiveness(self, obs, action, next_obs, state_changed):
        if obs not in self.effectiveness_counts:
            self.effectiveness_counts[obs] = {}

        if action not in self.effectiveness_counts[obs]:
            self.effectiveness_counts[obs][action] = 1
            return  # First time action is taken in this state

        transition = (obs, action, next_obs)

        is_novel_state = transition not in self.recent_transitions

        if state_changed and is_novel_state:
            self.effectiveness_counts[obs][action] += 1

    def calculate_bonus(self, obs, action):
        U = self.usage_counts[obs].get(action, 1)
        E = self.effectiveness_counts[obs].get(action, 0)
        term = (E ** self.H) / (U ** self.H)
        exp_term = self.eta ** (1 - term)
        bonus = (exp_term - 1) / (self.eta - 1)
        return bonus

    def update_state_visits(self, current_obs, next_obs):
        if current_obs not in self.state_visit_counts:
            self.state_visit_counts[current_obs] = 1

        if next_obs not in self.state_visit_counts:
            self.state_visit_counts[next_obs] = 0

        self.state_visit_counts[next_obs] += 1

    def calculate_intrinsic_reward(self, obs, action, next_obs, position_changed):
        # Penalize recently repeated transitions
        transition = (obs, action, next_obs)

        is_reward_available = position_changed and transition not in self.recent_transitions

        # If the agent has moved to a new position or the action is invalid, calculate intrinsic reward
        state_count = self.state_visit_counts[next_obs] ** self.tau
        action_bonus = self.calculate_bonus(obs, action)

        self.recent_transitions.append(transition)  # Track the new transition
        reward = 0.0
        if is_reward_available:
            intrinsic_reward = action_bonus / np.sqrt(state_count)
            reward = intrinsic_reward + reward
        else:
            decay_factor = np.exp(-0.1 * state_count)  # Adjust decay factor as needed
            intrinsic_reward = action_bonus * decay_factor / np.sqrt(state_count)
            reward = min(-abs(intrinsic_reward), -1e-2)
        # print(
        #     f"Transition: {transition}, Reward: {reward}, IsReward: {is_reward_available}, State Count: {state_count}, Action Bonus: {action_bonus}")
        return reward

    def reset_episode(self):
        self.usage_counts.clear()
        self.effectiveness_counts.clear()
        self.state_visit_counts.clear()
        self.recent_transitions.clear()
