import numpy as np


class DoWhaMIntrinsicReward:
    def __init__(self, eta, H, tau):
        self.eta = eta
        self.H = H
        self.tau = tau
        self.usage_counts = {}  # Tracks the usage of each action
        self.effectiveness_counts = {}  # Tracks the effectiveness of each action
        self.state_visit_counts = {}  # Tracks the number of times a state is visited in the current episode
        self.intrinsic_rewards = {}

    def update_usage(self, state, action):
        if state not in self.usage_counts:
            self.usage_counts[state] = {}

        if action not in self.usage_counts[state]:
            self.usage_counts[state][action] = 0

        self.usage_counts[state][action] += 1

    def update_effectiveness(self, action, state, next_state, state_changed):
        if state not in self.effectiveness_counts:
            self.effectiveness_counts[state] = {}

        if action not in self.effectiveness_counts[state]:
            self.effectiveness_counts[state][action] = 0

        if state_changed and self.effectiveness_counts[state][action] == 0 and self.state_visit_counts.get(next_state,
                                                                                                           0) == 0:
            self.effectiveness_counts[state][action] = 1

        if state_changed and self.effectiveness_counts[state][action] == 1:
            self.effectiveness_counts[state][action] = 0

    def calculate_action_bonus(self, state, next_state, action):
        U = self.usage_counts[state].get(action, 0)
        E = self.effectiveness_counts[state].get(action, 0)
        # print(f"Usage (U): {U}, Effectiveness (E): {E}")  # Debug output
        term = (E ** self.H) / (U ** self.H)
        exp_term = self.eta ** (1 - term)
        bonus = (exp_term - 1) / (self.eta - 1)
        # print(f"Term: {term}, Exp Term: {exp_term}, Bonus: {bonus}")  # Debug output
        return bonus

    def update_state_visits(self, state):
        self.state_visit_counts[state] = self.state_visit_counts.get(state, 0) + 1

    def calculate_intrinsic_reward(self, action, state, next_state):
        if state != next_state:
            state_count = self.state_visit_counts.get(next_state, 1) ** self.tau
            action_bonus = self.calculate_action_bonus(state, next_state, action)
            intrinsic_reward = action_bonus / np.sqrt(state_count)
            # print(f"State Count: {state_count}, Action Bonus: {action_bonus}, Intrinsic Reward: {intrinsic_reward}")

            # if state not in self.intrinsic_rewards:
            #     self.intrinsic_rewards[state] = []
            #
            # self.intrinsic_rewards[state].append(intrinsic_reward)

            # print(self.intrinsic_rewards[state])

            return intrinsic_reward
        return 0.0

    def reset_episode(self):
        self.state_visit_counts.clear()
