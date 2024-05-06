import numpy as np


class DoWhaMIntrinsicReward:
    def __init__(self, eta, H, tau):
        self.eta = eta
        self.H = H
        self.tau = tau
        self.usage_counts = {}  # Tracks the usage of each action
        self.effectiveness_counts = {}  # Tracks the effectiveness of each action
        self.state_visit_counts = {}  # Tracks the number of times a state is visited in the current episode

    def update_usage(self, action):
        self.usage_counts[action] = self.usage_counts.get(action, 0) + 1

    def update_effectiveness(self, action, state_changed):
        if state_changed:
            self.effectiveness_counts[action] = self.effectiveness_counts.get(action, 0) + 1

    def calculate_action_bonus(self, action):
        U = self.usage_counts.get(action, 1)  # Avoid division by zero
        E = self.effectiveness_counts.get(action, 0)
        bonus = (self.eta ** (1 - (E ** self.H) / (U ** self.H)) - 1) / (self.eta - 1)
        return bonus

    def update_state_visits(self, state):
        self.state_visit_counts[state] = self.state_visit_counts.get(state, 0) + 1

    def calculate_intrinsic_reward(self, action, state, next_state):
        if state != next_state:
            state_count = self.state_visit_counts.get(next_state, 1) ** self.tau
            action_bonus = self.calculate_action_bonus(action)
            intrinsic_reward = action_bonus / np.sqrt(state_count)
            return intrinsic_reward
        return 0.0

    def reset_episode(self):
        self.state_visit_counts.clear()
