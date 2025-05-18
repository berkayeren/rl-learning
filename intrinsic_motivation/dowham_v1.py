import numpy as np


class DoWhaMIntrinsicRewardV1:
    def __init__(self, eta, H, tau):
        self.eta = eta  # Decay parameter
        self.H = H  # Ignored in the original paper
        self.tau = tau  # State normalization exponent
        self.usage_counts = {}  # Tracks U(a): action usage counts per state
        self.effectiveness_counts = {}  # Tracks E(a): action effectiveness counts per state
        self.state_visit_counts = {}  # Tracks state visit counts per state

    def update_usage(self, obs, action):
        """
        Increment the usage count for the action in the current state.
        """
        if obs not in self.usage_counts:
            self.usage_counts[obs] = {}
        self.usage_counts[obs][action] = self.usage_counts[obs].get(action, 0) + 1

    def update_effectiveness(self, obs, action, next_obs, state_changed):
        """
        Increment the effectiveness count for the action if it changes the state.
        """
        if obs not in self.effectiveness_counts:
            self.effectiveness_counts[obs] = {}
        if action not in self.effectiveness_counts[obs]:
            self.effectiveness_counts[obs][action] = 0

        if state_changed:
            self.effectiveness_counts[obs][action] += 1

    def calculate_bonus(self, obs, action):
        U = self.usage_counts[obs].get(action, 1)
        E = self.effectiveness_counts[obs].get(action, 0)

        # Handle first-time effective actions
        if U == 1 and E == 1:
            return 1.0  # Maximum reward for first effectiveness

        ratio = E / U
        exp_term = self.eta ** (1 - ratio)
        bonus = (exp_term - 1) / (self.eta - 1)
        return bonus

    def update_state_visits(self, current_obs, next_obs):
        """
        Increment the visit count for the current and next states.
        """
        if current_obs not in self.state_visit_counts:
            self.state_visit_counts[current_obs] = 1
        if next_obs not in self.state_visit_counts:
            self.state_visit_counts[next_obs] = 0

        self.state_visit_counts[next_obs] += 1

    def calculate_intrinsic_reward(self, obs, action, next_obs, position_changed):
        """
        Calculate the intrinsic reward based on action effectiveness and state visitation.
        """
        # Reward any action that results in a state change
        if not position_changed:
            return 0

        # Normalize the reward by state visit counts
        state_count = self.state_visit_counts.get(next_obs, 1) ** self.tau
        action_bonus = self.calculate_bonus(obs, action)
        reward = action_bonus / np.sqrt(state_count)
        return reward

    def reset_episode(self):
        """
        Reset all episodic counts for a new episode.
        """
        self.usage_counts.clear()
        self.effectiveness_counts.clear()
        self.state_visit_counts.clear()
