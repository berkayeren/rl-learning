"""
count_based.py — Count-based exploration intrinsic reward.

This module implements the classic count-based exploration baseline (Bellemare
et al., 2016; Ostrovski et al., 2017). Equation references below (Eq. 2.x) follow
the count-based formulation used as a baseline in this project.

The agent maintains a visitation count ``N(s_t, a_t)`` of how many times each
state-action pair has been encountered. On every transition the count is
incremented (Eq. 2.8) and the intrinsic reward is the inverse square root of the
updated count (Eq. 2.9):

    N(s_t, a_t) ← N(s_t, a_t) + 1                  (Eq. 2.8)
    r_count = 1 / sqrt(N(s_t, a_t))                (Eq. 2.9)

Rarely-seen pairs therefore yield a large bonus that decays as the agent revisits
them, shifting behavior from exploration toward exploitation over training. As
with the other intrinsic methods, this reward is added to the sparse extrinsic
reward (scaled by β in ``trainer.py``).
"""

import math
from collections import defaultdict

import numpy as np


class Count:
    """Visitation-count table keyed by ``(state, action)`` pairs.

    Thin wrapper over a ``defaultdict(int)`` providing increment and lookup of
    the count ``N(s, a)``.
    """

    def __init__(self):
        self.counts = defaultdict(int)

    def increment(self, state, action):
        """Increment the visitation count for ``(state, action)`` (Eq. 2.8)."""
        self.counts[(state, action)] += 1

    def get_count(self, state, action):
        """Return the current visitation count ``N(state, action)``."""
        return self.counts[(state, action)]


class CountExploration:
    """Count-based intrinsic reward generator (Eq. 2.8–2.9).

    Tracks per-``(state, action)`` visitation counts and returns the inverse
    square-root bonus on each update.

    Args:
        env: The environment instance (kept for interface consistency with the
            other intrinsic-reward modules).
        gamma: Discount factor.
        epsilon: Exploration parameter.
        alpha: Learning-rate parameter.
    """

    def __init__(self, env, gamma=0.99, epsilon=0.1, alpha=0.1):
        self.counts = {}

    def update(self, state, action, reward, next_state):
        """Increment the count for ``(state, action)`` and return its bonus.

        Implements Eq. 2.8–2.9: bumps ``N(state, action)`` by one and returns
        ``1 / sqrt(N(state, action))``. ``state`` is converted to a tuple so it
        can be used as a dictionary key.

        Args:
            state: The current state (e.g. the agent's grid position).
            action: The action taken.
            reward: The extrinsic reward (not used in the count bonus).
            next_state: The resulting state (not used in the count bonus).

        Returns:
            float: The count-based intrinsic reward ``1 / sqrt(N)``.
        """
        tup = (tuple(state), action)

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        return 1 / math.sqrt(new_count)
