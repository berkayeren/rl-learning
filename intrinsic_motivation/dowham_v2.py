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
        print("DoWhaM V2 Intrinsic Reward Initialized")
        self.action_state = {}
        self.eta = eta
        self.H = H
        self.tau = tau
        self.usage_counts = {}
        self.max_steps = max_steps
        self.effectiveness_counts = {}
        self.state_visit_counts = {}
        self.recent_transitions = UniqueDeque(
            maxlen=max_steps // transition_divisor)  # Track recent state transitions
        self.unseen_positions = set()
        self.visited_positions = set()
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

        # self.update_state_transition(obs, action, state_changed)

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
        if current_obs not in self.state_visit_counts:
            self.state_visit_counts[current_obs] = 1

        if next_obs not in self.state_visit_counts:
            self.state_visit_counts[next_obs] = 0

        self.state_visit_counts[next_obs] += 1

    def update_state_transition(self, obs, action, state_changed):
        if obs not in self.action_state:
            self.action_state[obs] = {}

        if action not in self.action_state[obs]:
            self.action_state[obs][action] = False

        self.action_state[obs][action] = state_changed

    def calculate_intrinsic_reward(self, obs, action, next_obs, state_changed, curr_view, next_view, next_pos):
        """
        Calculate the intrinsic reward based on action effectiveness and state visitation.
        """
        # Check if the agent's new position was previously unseen
        was_unseen_position = next_pos in self.unseen_positions

        self.visited_positions.add(next_pos)

        curr_set = set(curr_view)
        next_set = set(next_view)
        newly_seen_set = next_set - curr_set - self.unseen_positions - self.visited_positions
        newly_seen = list(newly_seen_set)

        # Check if there are newly seen positions
        has_newly_seen = len(newly_seen) > 0

        self.unseen_positions.update(newly_seen if len(newly_seen) != 0 else curr_set)
        self.unseen_positions -= self.visited_positions

        # Give reward if:
        # 1. Action results in state change AND
        # 2. Either there are newly seen positions OR the agent moved to a previously unseen position
        if not state_changed:
            return 0

        # Calculate base action bonus
        action_bonus = self.calculate_bonus(obs, action)
        state_count = len(self.unseen_positions) ** self.tau

        if state_count == 0:
            state_count = 1  # Avoid division by zero

        base_reward = action_bonus / np.sqrt(state_count)
        total_reward = 0.0
        # Enhanced reward structure
        if has_newly_seen:
            # Give substantial bonus for reaching a subgoal (previously unseen position)
            subgoal_bonus = 10 * action_bonus  # Double the action bonus for subgoal achievement
            total_reward = base_reward + subgoal_bonus
        elif was_unseen_position:
            # Standard reward for discovering new positions
            total_reward = base_reward
        # elif self.usage_counts[obs].get(action, 1) > 1:
        #     # No reward if neither condition is met
        #     return -0.1

        return round(total_reward, 5)

    def reset_episode(self):
        self.state_visit_counts.clear()
        # self.usage_counts.clear()
        self.unseen_positions.clear()
        self.visited_positions.clear()
        # self.effectiveness_counts.clear()
        # self.recent_transitions.clear()
        # self.action_state.clear()
