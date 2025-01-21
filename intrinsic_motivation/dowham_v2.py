import collections
import random

import numpy as np


class DoWhaMIntrinsicRewardV2:
    def __init__(self, eta, H, tau, randomize_state_transition=False, max_steps=200):
        self.eta = eta
        self.H = H
        self.tau = tau
        self.usage_counts = {}
        self.effectiveness_counts = {}
        self.state_visit_counts = {}
        self.recent_transitions = collections.deque(maxlen=max_steps)  # Track recent state transitions
        self.randomize_state_transition = randomize_state_transition

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

        if self.randomize_state_transition:
            is_novel_state = random.choice([True, transition not in self.recent_transitions]),
        else:
            is_novel_state = transition not in self.recent_transitions

        if state_changed and is_novel_state:
            self.effectiveness_counts[obs][action] += 1

    def calculate_bonus(self, obs, action):
        U = self.usage_counts[obs].get(action, 1)
        E = self.effectiveness_counts[obs].get(action, 0)
        ratio = E / U
        term = ratio ** self.H
        exp_term = self.eta ** term
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
