import collections
import hashlib
import random

import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Key, Goal
from minigrid.envs import FourRoomsEnv

from intrinsic_motivation.count_based import CountExploration
from intrinsic_motivation.dowham import DoWhaMIntrinsicReward


def hash_dict(d):
    hashable_items = []
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            value = value.tobytes()
        hashable_items.append((key, value))
    return abs(hash(tuple(sorted(hashable_items))))


class CustomPlaygroundEnv(FourRoomsEnv):
    def __str__(self):
        if self.agent_pos is None:
            self.reset()

        return super().__str__()

    def __init__(self, intrinsic_reward_scaling=0.05, eta=40, H=1, tau=0.5, size=11, render_mode=None, **kwargs):
        self.agent_pos = (1, 1)
        self.goal_pos = (16, 16)
        self.agent_dir = 0
        self.success_rate: float = 0.0
        self.done = False
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

        self.minNumRooms = kwargs.pop('minNumRooms', 4)
        self.maxNumRooms = kwargs.pop('maxNumRooms', 4)
        self.maxRoomSize = kwargs.pop('maxRoomSize', 10)
        self.max_possible_rooms = kwargs.pop('max_possible_rooms', 6)

        super().__init__(
            max_steps=200, agent_view_size=size, render_mode=render_mode,
            agent_pos=self.agent_pos, goal_pos=self.goal_pos, tile_size=8,
            **kwargs
        )

        # self.spec = EnvSpec("CustomPlaygroundEnv-v0", max_episode_steps=200)
        self.success_history = collections.deque(maxlen=1024)

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

    def observations_changed(self, current_obs, next_obs):
        # Compare images
        images_equal = np.array_equal(current_obs['image'], next_obs['image'])
        # Compare directions
        directions_equal = current_obs['direction'] == next_obs['direction']
        # You can include mission if it changes over time
        # For MiniGrid, the mission usually remains the same
        return not (images_equal and directions_equal)

    def hash(self, size=32):
        """Compute a hash that uniquely identifies the current state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()
        grid, vis_mask = self.gen_obs_grid(agent_view_size=self.size)

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        to_encode = [image.tolist(), self.agent_pos, self.agent_dir]
        for item in to_encode:
            sample_hash.update(str(item).encode("utf8"))

        return sample_hash.hexdigest()[:size]

    def step(self, action):
        self.action_count[action] += 1
        current_state = self.hash()
        current_obs = self.gen_obs()

        obs, reward, done, info, _ = super().step(action)

        next_state = self.hash()
        next_obs = self.gen_obs()

        if self.enable_dowham_reward:
            self.dowham_reward.update_state_visits(current_state, next_state)
            state_changed = self.observations_changed(current_obs, next_obs)
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

        self.done = done

        if done:
            reward += max(10, 100 / np.sqrt(sum(self.action_count.values())))

        return obs, reward, done, info, {}

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
        self.success_history.append(self.done)
        self.success_rate = sum(self.success_history) / len(self.success_history)

        obs = super().reset(**kwargs)
        self.done = False
        obs = {
            'image': obs[0]['image'],
            'direction': np.array(self.agent_dir, dtype=np.int64),
            'mission': np.array([ord(c) for c in self.mission[:1]], dtype=np.uint8)  # Simplified mission representation
        }

        return obs, {}
