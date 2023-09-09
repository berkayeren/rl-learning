import os

import gym
from gym.spaces import Discrete


class Environment:
    action_space: gym.spaces.Space
    observation_space: gym.spaces.Space

    def __init__(self, *args, **kwargs):
        self.seeker, self.goal = (0, 0), (4, 4)
        self.info = {'seeker': self.seeker, 'goal': self.goal}

        self.action_space = Discrete(4)
        self.observation_space = Discrete(5 * 5)

    def reset(self):
        """Reset seeker position and return observations."""
        self.seeker = (0, 0)

        return self.get_observation()

    def get_observation(self):
        """Encode the seeker position as integer"""
        return 5 * self.seeker[0] + self.seeker[1]

    def get_reward(self):
        """Reward finding the goal"""
        return 1 if self.seeker == self.goal else 0

    def is_done(self):
        """We're done if we found the goal"""
        return self.seeker == self.goal

    def step(self, action):
        """Take a step in a direction and return all available information."""
        if action == 0:  # move down
            self.seeker = (min(self.seeker[0] + 1, 4), self.seeker[1])
        elif action == 1:  # move left
            self.seeker = (self.seeker[0], max(self.seeker[1] - 1, 0))
        elif action == 2:  # move up
            self.seeker = (max(self.seeker[0] - 1, 0), self.seeker[1])
        elif action == 3:  # move right
            self.seeker = (self.seeker[0], min(self.seeker[1] + 1, 4))
        else:
            raise ValueError("Invalid action")

        obs = self.get_observation()
        rew = self.get_reward()
        done = self.is_done()
        return obs, rew, done, self.info

    def render(self, *args, **kwargs):
        """Render the environment, e.g., by printing its representation."""
        os.system('cls' if os.name == 'nt' else 'clear')

        grid = [['| ' for _ in range(5)] + ["|\n"] for _ in range(5)]
        grid[self.goal[0]][self.goal[1]] = '|G'
        grid[self.seeker[0]][self.seeker[1]] = '|S'

        print(''.join([''.join(grid_row) for grid_row in grid]))


class GymEnvironment(Environment, gym.Env):
    def __init__(self, *args, **kwargs):
        """Make our original Environment a gym `Env`."""
        super().__init__(*args, **kwargs)
