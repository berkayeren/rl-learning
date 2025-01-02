import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RunningStats:
    def __init__(self, epsilon=1e-8):
        self.mean = 0.0
        self.var = 0.0
        self.count = epsilon  # Initialize with a small value to avoid division by zero

    def update(self, reward):
        """
        Update running mean and variance with the new reward.
        """
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var += delta * delta2

    def normalize(self, reward):
        """
        Normalize the reward using running mean and variance.
        """
        std = np.sqrt(self.var / self.count + 1e-8)  # Avoid division by zero
        return (reward - self.mean) / std

    @staticmethod
    def _update_stats(mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        total_count = count + batch_count

        new_mean = mean + delta * batch_count / total_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / total_count
        new_var = M2 / total_count
        new_count = total_count

        return new_mean, new_var, new_count


class TargetNetwork(nn.Module):
    """Random Target Network: Outputs fixed embeddings for observations."""

    def __init__(self, input_dim, embed_dim):
        super(TargetNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class PredictorNetwork(nn.Module):
    """Trainable Predictor Network: Learns to mimic the target network."""

    def __init__(self, input_dim, embed_dim):
        super(PredictorNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class RNDModule:
    """
    RND Module to calculate intrinsic rewards and train the predictor network.
    """

    def __init__(self, embed_dim=64, predictor_lr=1e-5, reward_scale=0.1):
        input_dim = 1447  # Hardcoded for the given environment

        # Target and Predictor Networks
        self.target_network = TargetNetwork(input_dim, embed_dim)
        self.predictor_network = PredictorNetwork(input_dim, embed_dim)

        # Freeze target network
        for param in self.target_network.parameters():
            param.requires_grad = False

        # Optimizer for the predictor network
        self.optimizer = optim.Adam(self.predictor_network.parameters(), lr=predictor_lr)

        # Running stats for normalization
        self.obs_normalizer = RunningStats()
        self.reward_normalizer = RunningStats()
        self.reward_scale = reward_scale

    def normalize_obs(self, observation):
        """
        Normalize the observation using running statistics.
        """
        normalized_obs = self.obs_normalizer.normalize(observation)
        return np.clip(normalized_obs, -5, 5)  # Clip to [-5, 5] for stability

    def compute_intrinsic_reward(self, observation):
        """
        Compute the intrinsic reward as the prediction error.
        """
        observation = numpy.array(observation)
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        target_output = self.target_network(observation)
        predictor_output = self.predictor_network(observation)
        intrinsic_reward = torch.mean((target_output - predictor_output) ** 2, dim=-1).item()

        # Normalize intrinsic reward
        normalized_reward = self.reward_normalizer.normalize(intrinsic_reward)
        return normalized_reward * self.reward_scale

    def update_predictor(self, observations):
        """
        Update the predictor network to minimize prediction error.
        """
        observations = numpy.array(observations)
        observations = torch.tensor(observations, dtype=torch.float32)
        target_output = self.target_network(observations)
        predictor_output = self.predictor_network(observations)

        # Compute distillation loss
        loss = torch.mean((target_output - predictor_output) ** 2)

        # Optimize predictor network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_obs_normalizer(self, observation):
        """
        Update the observation normalization parameters.
        """
        self.obs_normalizer.update(observation)

    def update_reward_normalizer(self, intrinsic_rewards):
        """
        Update the reward normalization parameters.
        """
        self.reward_normalizer.update(intrinsic_rewards)
