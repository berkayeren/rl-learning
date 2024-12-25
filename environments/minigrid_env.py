import collections
import hashlib
import random

import numpy as np
from gymnasium.envs.registration import EnvSpec
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv

from intrinsic_motivation.count_based import CountExploration
from intrinsic_motivation.dowham_v1 import DoWhaMIntrinsicRewardV1
from intrinsic_motivation.dowham_v2 import DoWhaMIntrinsicRewardV2
from intrinsic_motivation.rnd import RNDModule


def hash_dict(d):
    hashable_items = []
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            value = value.tobytes()
        hashable_items.append((key, value))
    return abs(hash(tuple(sorted(hashable_items))))


class RewardNormalizer:
    def __init__(self, min_val=-1, max_val=1):
        self.min_val = min_val
        self.max_val = max_val
        self.running_min = float("inf")
        self.running_max = float("-inf")

    def normalize(self, reward):
        # Update the running min and max
        self.running_min = min(self.running_min, reward)
        self.running_max = max(self.running_max, reward)

        # Normalize using the observed range
        if self.running_max > self.running_min:  # Avoid division by zero
            normalized = (reward - self.running_min) / (self.running_max - self.running_min)
            return self.min_val + normalized * (self.max_val - self.min_val)
        else:
            return 0  # If no range yet, return 0


class CustomPlaygroundEnv(MiniGridEnv):

    @staticmethod
    def _gen_mission():
        return "reach the goal"

    def __str__(self):
        if self.agent_pos is None:
            self.reset()

        return super().__str__()

    def __init__(self, intrinsic_reward_scaling=1, eta=40, H=1, tau=0.5, size=11, render_mode=None, **kwargs):
        self.consider_position = kwargs.pop('consider_position', True)
        self.agent_pos = (1, 1)
        self.goal_pos = (16, 16)
        self.agent_dir = 0
        self.success_rate: float = 0.0
        self.done = False
        self.intrinsic_reward_scaling = intrinsic_reward_scaling
        self.normalizer = RewardNormalizer(0, 1)
        self.enable_dowham_reward_v1 = kwargs.pop('enable_dowham_reward_v1', False)
        self.enable_dowham_reward_v2 = kwargs.pop('enable_dowham_reward_v2', False)
        self.enable_count_based = kwargs.pop('enable_count_based', False)
        self.enable_rnd = kwargs.pop('enable_rnd', False)
        self.action_count = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
        }

        self.episode_history = []

        self.minNumRooms = kwargs.pop('minNumRooms', 4)
        self.maxNumRooms = kwargs.pop('maxNumRooms', 4)
        self.maxRoomSize = kwargs.pop('maxRoomSize', 10)
        self.max_possible_rooms = kwargs.pop('max_possible_rooms', 6)
        self.env_name = kwargs.pop('env_name', "CustomPlaygroundEnv-v0")
        self.spec = EnvSpec(self.env_name, max_episode_steps=200)
        self.size = 19
        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            max_steps=200, render_mode=render_mode, tile_size=8,
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            **kwargs
        )

        self.max_steps = 200
        self.success_history = collections.deque(maxlen=100)

        if self.enable_dowham_reward_v1:
            print("Enabling DoWhaM intrinsic reward v1")
            self.dowham_reward = DoWhaMIntrinsicRewardV1(eta, H, tau)
            self.intrinsic_reward = 0.0

        if self.enable_dowham_reward_v2:
            print("Enabling DoWhaM intrinsic reward with Negative Reward")
            self.dowham_reward = DoWhaMIntrinsicRewardV2(eta, H, tau)
            self.intrinsic_reward = 0.0
            self.normalizer = RewardNormalizer(-1, 1)

        if self.enable_count_based:
            print("Enabling count-based exploration")
            self.count_exploration = CountExploration(self, gamma=0.99, epsilon=0.1, alpha=0.1)
            self.count_bonus = 0.0

        if self.enable_rnd:
            print("Enabling Random Network Distillation")
            self.rnd = RNDModule(observation_space=self.observation_space, embed_dim=64,
                                 reward_scale=10)

        self.states = np.full((self.width, self.height), 0)
        self.total_episode_reward = 0.0

    @staticmethod
    def _gen_mission():
        return "traverse the rooms to get to the goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        # Place a vertical wall to divide the grid into two halves
        self.grid.vert_wall(width // 2, 0, height)

        # Place a horizontal wall to divide the grid into two halves
        self.grid.horz_wall(0, height // 2, width)
        # Clear space for doors in the walls
        self.grid.set(width // 2, height // 4, None)  # Clear space for door in vertical wall (upper)
        self.grid.set(width // 2, 3 * height // 4, None)  # Clear space for door in vertical wall (lower)
        self.grid.set(width // 4, height // 2, None)  # Clear space for door in horizontal wall (left)
        self.grid.set(3 * width // 4, height // 2, None)  # Clear space for door in horizontal wall (right)

        self.agent_pos = (1, 1)
        self.put_obj(Goal(), 16, 16)

        self.agent_dir = random.randint(0, 3)
        self.mission = "traverse the rooms to get to the goal"

    def observations_changed(self, current_obs, next_obs):
        # Compare images
        images_equal = current_obs.get('position') == next_obs.get('position')
        directions_equal = True
        if self.consider_position:
            # Compare directions
            directions_equal = current_obs['direction'] == next_obs['direction']

        # You can include mission if it changes over time
        # For MiniGrid, the mission usually remains the same
        return not (images_equal and directions_equal)

    def hash(self, size=32):
        """Compute a hash that uniquely identifies the full state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()

        # Get the full grid representation
        full_grid = self.grid.encode()  # Encodes the entire grid into a NumPy array

        # Convert the full grid to a list for hashing
        full_grid_list = full_grid.tolist()

        # Include agent's position and direction
        to_encode = [full_grid_list, self.agent_pos, self.agent_dir]

        # Update the hash with each element
        for item in to_encode:
            sample_hash.update(str(item).encode("utf8"))

        # Return the hashed value
        return sample_hash.hexdigest()[:size]

    def gen_obs(self):
        full_grid = self.grid.encode()
        return {
            'image': full_grid,
            'direction': self.agent_dir,
            'position': self.agent_pos,
        }

    def flatten_observation(self, observation):
        grid_flattened = observation["image"].flatten()
        position_flattened = np.array(self.agent_pos, dtype=np.float32)
        direction_flattened = np.array([self.agent_dir], dtype=np.float32)

        # Concatenate all components into a single flat array
        return np.concatenate([grid_flattened, position_flattened, direction_flattened])

    def step(self, action):
        self.states[self.agent_pos[0]][self.agent_pos[1]] += 1
        self.action_count[action] += 1
        current_state = self.hash()
        current_obs = self.gen_obs()
        obs, reward, done, info, _ = super().step(action)

        next_state = self.hash()
        next_obs = self.gen_obs()
        self.intrinsic_reward = 0.0

        if self.enable_dowham_reward_v1 or self.enable_dowham_reward_v2:
            self.dowham_reward.update_state_visits(current_state, next_state)
            state_changed = self.observations_changed(current_obs, next_obs)
            self.dowham_reward.update_usage(current_state, action)
            self.dowham_reward.update_effectiveness(current_state, action, next_state, state_changed)
            intrinsic_reward = self.dowham_reward.calculate_intrinsic_reward(current_state, action, next_state,
                                                                             state_changed)
            self.intrinsic_reward = self.intrinsic_reward_scaling * intrinsic_reward

        if self.enable_count_based:
            self.intrinsic_reward = self.count_exploration.update(current_state, action, reward, next_state)
            self.intrinsic_reward = self.intrinsic_reward_scaling * self.intrinsic_reward

        # Compute intrinsic reward if RND is active
        if self.enable_rnd:
            flat_obs = self.flatten_observation(next_obs)
            self.intrinsic_reward = self.rnd.compute_intrinsic_reward(flat_obs)

            # Update predictor network
            self.rnd.update_predictor([flat_obs])

        self.intrinsic_reward = self.normalizer.normalize(self.intrinsic_reward)
        reward += self.intrinsic_reward
        self.done = done

        if done:
            total_size = self.width * self.height
            # Calculate the number of unique states visited by the agent
            unique_states_visited = np.count_nonzero(self.states)

            # Calculate the percentage of the environment the agent has visited
            percentage_visited = (unique_states_visited / total_size) * 100

            reward += max(10, int((percentage_visited * self.total_episode_reward) / 100))

        return obs, reward, done, info, {}

    def reset(self, **kwargs):
        self.episode_history = []
        if self.enable_dowham_reward_v1 or self.enable_dowham_reward_v2:
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
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        obs, _ = super().reset(**kwargs)

        # Update observation normalization if RND is active
        if self.enable_rnd:
            self.rnd.update_obs_normalizer(self.flatten_observation(obs))

        self.states = np.full((self.width, self.height), 0)
        self.done = False
        self.total_episode_reward = 0.0

        return obs, {}
