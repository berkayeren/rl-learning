import collections
import random
from collections import defaultdict

import numpy as np
from gymnasium.envs.registration import EnvSpec
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Door, Key
from minigrid.envs import MultiRoomEnv
from gymnasium.spaces import Box, Dict, Discrete
from minigrid.core.mission import MissionSpace


def hash_dict(d):
    hashable_items = []
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            value = value.tobytes()
        hashable_items.append((key, value))
    return abs(hash(tuple(sorted(hashable_items))))


import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CustomPlaygroundEnv(MultiRoomEnv):
    def __str__(self):
        if self.agent_pos is None:
            self.reset()

        return super().__str__()

    def __init__(self, intrinsic_reward_scaling=0.85, eta=40, H=1, tau=0.5, size=7, render_mode=None,
                 prediction_net=None,
                 prediction_criterion=None,
                 prediction_optimizer=None,
                 enable_prediction_reward=False,
                 **kwargs):

        self.enable_prediction_reward = enable_prediction_reward
        if enable_prediction_reward:
            self.prediction_prob = 0.0
            self.prediction_reward = 0.0
            self.prediction_net = prediction_net
            self.prediction_criterion = prediction_criterion
            self.prediction_optimizer = prediction_optimizer

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

        super().__init__(minNumRooms=4, maxNumRooms=4, max_steps=200, render_mode=render_mode)

        # self.carrying = Key('yellow')
        self.spec = EnvSpec("CustomPlaygroundEnv-v0", max_episode_steps=200)

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

    def preprocess_observation(self, obs):
        image = torch.tensor(obs["image"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        direction = torch.tensor([[obs["direction"]]], dtype=torch.float32)
        return image, direction

    def predict_action_with_probability(self, observation):
        image, direction = self.preprocess_observation(observation)

        with torch.no_grad():
            logits = self.prediction_net(image, direction)

            # Apply softmax to convert logits to probabilities
            probabilities = F.softmax(logits, dim=1)

            # Get the predicted action (index of max probability)
            predicted_action = torch.argmax(probabilities, dim=1).item()

            # Get the probability of the predicted action
            predicted_probability = probabilities[0, predicted_action].item()

        return predicted_action, predicted_probability

    def get_all_action_probabilities(self, observation):
        image, direction = self.preprocess_observation(observation)
        image = image.unsqueeze(0)
        direction = direction.unsqueeze(0)

        with torch.no_grad():
            logits = self.prediction_net(image, direction)
            probabilities = F.softmax(logits, dim=1)

        return probabilities[0].tolist()  # Convert to list for easy handling

    def step(self, action):
        self.action_count[action] += 1
        current_state = self.agent_pos, self.agent_dir
        current_obs = hash_dict(self.gen_obs())

        if self.enable_prediction_reward:
            initial_observation = self.gen_obs()

        obs, reward, done, info, _ = super().step(action)

        if self.enable_prediction_reward:
            next_observation = self.gen_obs()

        next_state = self.agent_pos, self.agent_dir
        next_obs = hash_dict(self.gen_obs())

        if self.enable_dowham_reward:
            self.dowham_reward.update_state_visits(current_state, next_state)
            state_changed = current_state[0] != next_state[0]
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

        if self.enable_prediction_reward:
            self.prediction_reward, predicted_action, self.prediction_prob = self.prediction_error(action,
                                                                                                   initial_observation)

            self.episode_history.append(
                {"current_obs": initial_observation, "agent_dir": self.agent_dir, "action": action, "reward": reward,
                 "next_obs": next_observation, "prediction_reward": self.prediction_reward,
                 "predicted_action": predicted_action,
                 "prob": self.prediction_prob})
            # Add the prediction-based reward to the total reward
            reward += self.prediction_reward

        return obs, reward, done, info, {}

    def prediction_error(self, action, initial_observation):
        predicted_action, prob = self.predict_action_with_probability(initial_observation)
        # print(f"Action:{action}, Predicted action: {predicted_action}, with probability: {prob:.4f}")
        # Prediction-based reward shaping
        prediction_reward_scale = 0.3  # Adjust this value to control the impact of the prediction reward
        if action == predicted_action:
            prediction_reward = prediction_reward_scale * prob
            # print(f"Action matches prediction. Bonus reward: {prediction_reward:.4f}")
        else:
            prediction_reward = -prediction_reward_scale * (1 - prob)
            # print(f"Action doesn't match prediction. Penalty: {prediction_reward:.4f}")
        # Exploration encouragement
        exploration_threshold = 0.3  # Adjust this value based on your needs
        exploration_bonus_scale = 0.05  # Adjust this value to control the impact of the exploration bonus
        if prob < exploration_threshold and action != predicted_action:
            exploration_bonus = exploration_bonus_scale * (1 - prob)
            # print(f"Exploration bonus: {exploration_bonus:.4f}")
            prediction_reward += exploration_bonus
        return prediction_reward, predicted_action, prob

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

        obs = super().reset(**kwargs)
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

        if state_changed and self.is_novel_state(next_obs):
            self.effectiveness_counts[obs][action] += 1

    def is_novel_state(self, next_obs):
        if next_obs not in self.state_visit_counts:
            self.state_visit_counts[next_obs] = 1
            return True
        else:
            self.state_visit_counts[next_obs] += 1
            return False

    def calculate_bonus(self, obs, action):
        # if obs not in self.usage_counts or obs not in self.effectiveness_counts:
        #     return 0

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
        reward = 0.0

        # If the agent has moved to a new position or the action is invalid, calculate intrinsic reward
        state_count = self.state_visit_counts[next_obs] ** self.tau
        action_bonus = self.calculate_bonus(obs, action)

        if position_changed:
            intrinsic_reward = action_bonus / np.sqrt(state_count)
            return intrinsic_reward + reward
        else:
            intrinsic_reward = action_bonus / np.sqrt(state_count)
            return -abs(intrinsic_reward)

    def reset_episode(self):
        self.usage_counts.clear()
        self.effectiveness_counts.clear()
        self.state_visit_counts.clear()
