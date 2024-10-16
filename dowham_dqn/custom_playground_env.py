import random
from collections import defaultdict

import numpy as np
from gymnasium.envs.registration import EnvSpec
from gymnasium.spaces import Box, Dict, Discrete
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Door, Key
from minigrid.envs import MultiRoomEnv


def hash_dict(d):
    hashable_items = []
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            value = value.tobytes()
        hashable_items.append((key, value))
    return abs(hash(tuple(sorted(hashable_items))))


import torch
import torch.nn as nn
import torch.nn.functional as F

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# Initialize Qdrant client
qdrant_client = QdrantClient(host='localhost', port=6333)

# Define the vector size (dimensionality of your state vectors)
VECTOR_SIZE = 150

COLLECTION_NAME = 'state_vectors'

# Check if the collection exists
try:
    qdrant_client.get_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' already exists.")
except Exception as e:
    print(f"Collection '{COLLECTION_NAME}' does not exist. Creating a new one.")
    # Create a collection for states
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=rest.VectorParams(size=VECTOR_SIZE, distance=rest.Distance.COSINE)
    )
    print(f"Collection '{COLLECTION_NAME}' created.")


class MiniGridNet(nn.Module):
    def __init__(self):
        super(MiniGridNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7 + 1, 128)
        self.fc2 = nn.Linear(128, 7)  # 7 possible actions

    def forward(self, x, direction):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, direction), dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def get_similar_states(state_vector, top_k=5):
    search_result = qdrant_client.search(
        collection_name='state_vectors',
        query_vector=state_vector.tolist(),
        limit=top_k
    )
    return search_result


class CustomPlaygroundEnv(MultiRoomEnv):
    def __init__(self, intrinsic_reward_scaling=0.05, eta=40, H=1, tau=0.5, size=7, render_mode=None,
                 prediction_net=None,
                 prediction_criterion=None,
                 prediction_optimizer=None,
                 enable_prediction_reward=False,
                 **kwargs):
        self.novelty_score = 0.0
        self.enable_prediction_reward = enable_prediction_reward
        self.prediction_prob = 0.0
        self.prediction_reward = 0.0
        self.prediction_net = prediction_net
        self.prediction_criterion = prediction_criterion
        self.prediction_optimizer = prediction_optimizer

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

        super().__init__(minNumRooms=4, maxNumRooms=4, max_steps=200, agent_view_size=size, render_mode=render_mode)

        # Define the observation space to include image, direction, and mission
        self.observation_space = Dict({
            'image': Box(low=0, high=255, shape=(size, size, 3), dtype=np.uint8),
            'direction': Discrete(4),
            'mission': Box(low=0, high=255, shape=(1,), dtype=np.uint8)  # Simplified mission space for demonstration
        })
        # self.carrying = Key('yellow')
        self.spec = EnvSpec("CustomPlaygroundEnv-v0", max_episode_steps=200)

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
        self.put_obj(Door('yellow'), width // 2, height // 4)  # Door in the upper part of the vertical wall
        self.put_obj(Door('yellow'), width // 2, 3 * height // 4)  # Door in the lower part of the vertical wall
        self.put_obj(Door('yellow'), width // 4, height // 2)  # Door in the left part of the horizontal wall
        self.put_obj(Door('yellow'), 3 * width // 4, height // 2)  # Door in the right part of the horizontal wall
        self.put_obj(Key('yellow'), 3, 3)
        self.agent_pos = (1, 1)
        self.put_obj(Goal(), 14, 7)

        self.agent_dir = random.randint(0, 3)
        self.mission = "traverse the rooms to get to the goal"

    def preprocess_observation(self, obs):
        image = torch.tensor(obs["image"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        direction = torch.tensor([[obs["direction"]]], dtype=torch.float32)
        return image, direction

    def predict_action_with_probability(self, observation):
        image, direction = self.preprocess_observation(observation)

        with torch.no_grad():
            logits = self.prediction_net(image, direction)

            # Apply softmax to convert logits to probabilities
            probabilities = F.softmax(logits, dim=1)

            # Get the predicted action (index of max probability)
            predicted_action = torch.argmax(probabilities, dim=1).item()

            # Get the probability of the predicted action
            predicted_probability = probabilities[0, predicted_action].item()

        return predicted_action, predicted_probability

    def get_all_action_probabilities(self, observation):
        image, direction = self.preprocess_observation(observation)
        image = image.unsqueeze(0)
        direction = direction.unsqueeze(0)

        with torch.no_grad():
            logits = self.prediction_net(image, direction)
            probabilities = F.softmax(logits, dim=1)

        return probabilities[0].tolist()  # Convert to list for easy handling

    def step(self, action):
        self.action_count[action] += 1
        current_state = self.agent_pos
        current_obs = self.hash()
        initial_observation = self.gen_obs()
        obs, reward, done, info, _ = super().step(action)
        next_observation = self.gen_obs()
        next_state = self.agent_pos
        next_obs = self.hash()

        self.novelty_score, visit_count, state_vector = self.intrinsic_reward_similarity_score(obs)

        if self.enable_dowham_reward:
            self.dowham_reward.update_state_visits(current_obs, next_obs)
            state_changed = current_state != next_state
            self.dowham_reward.update_usage(current_obs, action)
            self.dowham_reward.update_effectiveness(current_obs, action, next_obs, state_changed)
            intrinsic_reward = self.dowham_reward.calculate_intrinsic_reward(current_obs, action, next_obs,
                                                                             state_changed, visit_count)

            novelty_weight = max(0.1,
                                 1.0 - visit_count / (visit_count + 1))  # Decay novelty weight as visit count increases
            self.intrinsic_reward = novelty_weight * self.novelty_score + self.intrinsic_reward_scaling * intrinsic_reward
            reward += self.intrinsic_reward
            self.upsert_observation(action, self.novelty_score, reward, intrinsic_reward, state_vector, visit_count,
                                    next_state[0],
                                    next_state[1])

        if self.enable_count_based:
            bonus = self.count_exploration.update(current_obs, action, reward, next_obs)
            self.count_bonus = bonus
            reward += bonus

        obs = {
            'image': obs['image'],
            'direction': np.array(self.agent_dir, dtype=np.int64),
            'mission': np.array([ord(c) for c in self.mission[:1]], dtype=np.uint8)
        }

        if self.enable_prediction_reward:
            self.prediction_reward, predicted_action, self.prediction_prob = self.prediction_error(action,
                                                                                                   initial_observation)

            self.episode_history.append(
                {"current_obs": initial_observation, "agent_dir": self.agent_dir, "action": action, "reward": reward,
                 "next_obs": next_observation, "prediction_reward": self.prediction_reward,
                 "predicted_action": predicted_action,
                 "prob": self.prediction_prob})
            # Add the prediction-based reward to the total reward
            reward += self.prediction_reward

        return obs, reward, done, info, {}

    def state_to_vector(self, state, x, y):
        # Flatten the image
        image = state['image'].flatten()
        # Get direction
        direction = np.array([state['direction']])

        position = np.array([x, y])
        # Concatenate all components
        state_vector = np.concatenate([image, direction, position])
        # Normalize the vector
        norm = np.linalg.norm(state_vector)
        if norm > 0:
            state_vector = state_vector / norm
        return [float(x) for x in state_vector]

    def intrinsic_reward_similarity_score(self, next_obs):
        x, y = self.agent_pos  # Current agent position

        # Convert state to vector
        state_vector = self.state_to_vector(next_obs, x, y)

        # Query the vector database for similar states with filters
        similar_states = qdrant_client.search(
            collection_name='state_vectors',
            query_vector=state_vector,
            limit=1,  # Retrieve the most similar state
            search_params=rest.SearchParams(
                hnsw_ef=128  # Adjust ef parameter if needed
            ),
        )
        visit_count = 0
        # Compute the intrinsic reward based on similarity
        if similar_states:
            similarity_score = similar_states[0].score
            visit_count = similar_states[0].payload.get('visit_count', 0)
            similarity_score = np.clip(similarity_score, -1.0, 1.0)
            novelty_score = 1 - similarity_score
        else:
            novelty_score = 1.0  # Max novelty if no similar states are found

        return float(novelty_score), visit_count, state_vector

    def upsert_observation(self, action, novelty_score, reward, intrinsic_reward, state_vector, visit_count, x, y):
        qdrant_client.upsert(
            collection_name='state_vectors',
            points=[
                rest.PointStruct(
                    id=int(self.hash(), 16),
                    vector=state_vector,
                    payload={
                        'action': int(action),
                        'reward': float(reward),
                        'intrinsic_reward': intrinsic_reward,
                        'novelty_score': float(novelty_score),
                        'obs': str(self.hash()),
                        'visit_count': int(visit_count) + 1,
                        'x': int(x),
                        'y': int(y)
                    }
                )
            ]
        )

    def prediction_error(self, action, initial_observation):
        predicted_action, prob = self.predict_action_with_probability(initial_observation)
        # print(f"Action:{action}, Predicted action: {predicted_action}, with probability: {prob:.4f}")
        # Prediction-based reward shaping
        prediction_reward_scale = 0.3  # Adjust this value to control the impact of the prediction reward
        if action == predicted_action:
            prediction_reward = prediction_reward_scale * prob
            # print(f"Action matches prediction. Bonus reward: {prediction_reward:.4f}")
        else:
            prediction_reward = -prediction_reward_scale * (1 - prob)
            # print(f"Action doesn't match prediction. Penalty: {prediction_reward:.4f}")
        # Exploration encouragement
        exploration_threshold = 0.3  # Adjust this value based on your needs
        exploration_bonus_scale = 0.05  # Adjust this value to control the impact of the exploration bonus
        if prob < exploration_threshold and action != predicted_action:
            exploration_bonus = exploration_bonus_scale * (1 - prob)
            # print(f"Exploration bonus: {exploration_bonus:.4f}")
            prediction_reward += exploration_bonus
        return prediction_reward, predicted_action, prob

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

        obs = super().reset(**kwargs)
        obs = {
            'image': obs[0]['image'],
            'direction': np.array(self.agent_dir, dtype=np.int64),
            'mission': np.array([ord(c) for c in self.mission[:1]], dtype=np.uint8)  # Simplified mission representation
        }

        return obs, {}


class Count:
    def __init__(self):
        self.counts = defaultdict(int)

    def increment(self, state, action):
        self.counts[(state, action)] += 1

    def get_count(self, state, action):
        return self.counts[(state, action)]


class CountExploration:
    def __init__(self, env, gamma=0.99, epsilon=0.1, alpha=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.count = Count()
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))

    def update(self, state, action, reward, next_state):
        self.count.increment(state, action)
        count = self.count.get_count(state, action)
        bonus = 1.0 / np.sqrt(count)
        self.q_table[state][action] += self.alpha * (
                reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action] + bonus)
        return bonus

    def reset(self):
        self.count = Count()
        self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))


class DoWhaMIntrinsicReward:
    def __init__(self, eta, H, tau):
        self.eta = eta
        self.H = H
        self.tau = tau
        self.usage_counts = {}
        self.effectiveness_counts = {}
        self.state_visit_counts = {}

    def update_usage(self, obs, action):
        if obs not in self.usage_counts:
            self.usage_counts[obs] = {}
        self.usage_counts[obs][action] = self.usage_counts[obs].get(action, 0) + 1

    def update_effectiveness(self, obs, action, next_obs, state_changed):
        if obs not in self.effectiveness_counts:
            self.effectiveness_counts[obs] = {}

        if action not in self.effectiveness_counts[obs]:
            self.effectiveness_counts[obs][action] = 0

        if state_changed or obs != next_obs:
            self.effectiveness_counts[obs][action] += 1

    def calculate_bonus(self, obs, action):
        if obs not in self.usage_counts or obs not in self.effectiveness_counts:
            return 0

        U = self.usage_counts[obs].get(action, 1)
        E = self.effectiveness_counts[obs].get(action, 0)
        term = (E ** self.H) / (U ** self.H)
        exp_term = self.eta ** (1 - term)
        bonus = (exp_term - 1) / (self.eta - 1)
        return bonus

    def update_state_visits(self, current_obs, next_obs):
        if current_obs not in self.state_visit_counts:
            self.state_visit_counts[current_obs] = 0

        if next_obs not in self.state_visit_counts:
            self.state_visit_counts[next_obs] = 0

        self.state_visit_counts[next_obs] += 1

    def calculate_intrinsic_reward(self, obs, action, next_obs, position_changed, visit_count=0):
        reward = 0.0

        is_valid_action = False
        if action not in [0, 1, 2] and obs == next_obs:
            is_valid_action = True

        # If the agent has moved to a new position or the action is invalid, calculate intrinsic reward
        if position_changed or is_valid_action:
            state_count = visit_count ** self.tau
            action_bonus = self.calculate_bonus(obs, action)
            intrinsic_reward = action_bonus / np.sqrt(state_count)
            return intrinsic_reward + reward
        return 0.0

    def reset_episode(self):
        self.usage_counts.clear()
        self.effectiveness_counts.clear()
        self.state_visit_counts.clear()
