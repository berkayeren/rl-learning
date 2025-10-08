import math
from collections import defaultdict

import numpy as np


class Count:
    def __init__(self):
        self.counts = defaultdict(int)

    def increment(self, state, action):
        self.counts[(state, action)] += 1

    def get_count(self, state, action):
        return self.counts[(state, action)]


class CountExploration:
    def __init__(self, env, gamma=0.99, epsilon=0.1, alpha=0.1):
        self.counts = {}

    def update(self, state, action, reward, next_state):
        tup = (tuple(state), action)

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        return 1 / math.sqrt(new_count)
