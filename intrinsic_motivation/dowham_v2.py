from collections import deque

import numpy as np


class UniqueDeque(deque):
    """
    A deque that ensures all elements are unique.
    When adding elements, duplicates are not added again.
    """

    def __init__(self, *args, maxlen=None, **kwargs):
        super().__init__(*args, maxlen=maxlen)
        self._set = set(self)  # Internal set to enforce uniqueness

    def append(self, item):
        """Add an item to the right end of the deque if it is not already present."""
        if item not in self._set:
            super().append(item)
            self._set.add(item)

    def appendleft(self, item):
        """Add an item to the left end of the deque if it is not already present."""
        if item not in self._set:
            super().appendleft(item)
            self._set.add(item)

    def extend(self, iterable):
        """Extend the deque by appending elements from the iterable if they are not already present."""
        for item in iterable:
            self.append(item)

    def extendleft(self, iterable):
        """Extend the deque by appending elements to the left from the iterable if they are not already present."""
        for item in iterable:
            self.appendleft(item)

    def remove(self, item):
        """Remove the first occurrence of the item."""
        try:
            super().remove(item)
        except ValueError:
            pass
        try:
            self._set.remove(item)
        except KeyError:
            pass

    def pop(self):
        """Remove and return an element from the right end of the deque."""
        item = super().pop()
        self._set.remove(item)
        return item

    def popleft(self):
        """Remove and return an element from the left end of the deque."""
        item = super().popleft()
        self._set.remove(item)
        return item

    def clear(self):
        """Clear all items from the deque."""
        super().clear()
        self._set.clear()

    def __contains__(self, item):
        """Check if an item is in the deque."""
        return item in self._set


class DoWhaMIntrinsicRewardV2:
    def __init__(self, eta, H, tau, randomize_state_transition=False, max_steps=200, transition_divisor=1):
        self.action_state = {}
        self.eta = eta
        self.H = H
        self.tau = tau
        self.usage_counts = {}
        self.max_steps = max_steps
        self.effectiveness_counts = {}
        self.state_visit_counts = {}
        self._state_visit_counts = {}
        self.recent_transitions = UniqueDeque(
            maxlen=max_steps // transition_divisor)  # Track recent state transitions
        self.randomize_state_transition = randomize_state_transition

    def update_usage(self, obs, action):
        if obs not in self.usage_counts:
            self.usage_counts[obs] = {}

        if action not in self.usage_counts[obs]:
            self.usage_counts[obs][action] = 0

        self.usage_counts[obs][action] += 1

    def update_effectiveness(self, obs, action, next_obs, state_changed):
        if obs not in self.effectiveness_counts:
            self.effectiveness_counts[obs] = {}

        if action not in self.effectiveness_counts[obs]:
            self.effectiveness_counts[obs][action] = 0

        if state_changed:
            self.effectiveness_counts[obs][action] += 1

        self.update_state_transition(obs, action, state_changed)

    def _calculate_bonus(self, obs, action):
        U = self.usage_counts[obs].get(action, 1)
        E = self.effectiveness_counts[obs].get(action, 0)
        ratio = E / U
        term = ratio ** self.H
        exp_term = self.eta ** term
        bonus = (exp_term - 1) / (self.eta - 1)
        return bonus

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
        self._state_visit_counts[next_obs] = self._state_visit_counts.get(next_obs, 0) + 1

        if current_obs not in self.state_visit_counts:
            self.state_visit_counts[current_obs] = 1

        if next_obs not in self.state_visit_counts:
            self.state_visit_counts[next_obs] = 0

        self.state_visit_counts[next_obs] += 1

    def update_state_transition(self, obs, action, state_changed):
        if obs not in self.action_state:
            self.action_state[obs] = {}

        if action not in self.action_state[obs]:
            self.action_state[obs][action] = []

        self.action_state[obs][action].append(state_changed)

    def calculate_intrinsic_reward(self, obs, action, next_obs, state_changed):
        # If the agent has moved to a new position or the action is invalid, calculate intrinsic reward
        state_count = self.state_visit_counts[obs]
        action_bonus = self.calculate_bonus(obs, action)
        intrinsic_reward = action_bonus / np.sqrt(state_count)
        reward = 0.0

        if state_changed:
            reward = intrinsic_reward
        elif self.action_state[obs][action].count(False) > 1:
            penalty = 1 - (1 / (self.action_state[obs][action].count(False) / np.sqrt(state_count)))
            reward = max(-penalty, -1.0)  # Cap the penalty at -1.0

        return round(reward, 5)

    def reset_episode(self):
        self.state_visit_counts.clear()
        self.recent_transitions.clear()
        self.action_state.clear()
