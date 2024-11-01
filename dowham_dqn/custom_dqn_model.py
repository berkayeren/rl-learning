from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import Tensor


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class MinigridPolicyNet(TorchModelV2, nn.Module):
    def __init__(self, obs_space: Space, action_space: Space, num_outputs: int, model_config: Dict, name: str):
        """
        Initialize the CustomMinigridPolicyNet.

        Args:
            obs_space (Space): The observation space of the environment.
            action_space (Space): The action space of the environment.
            num_outputs (int): The number of outputs the network should have.
            model_config (Dict): The configuration for the model.
            name (str): The name of the model.
        """
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_shape = obs_space.shape
        self.num_actions = action_space.n

        # Convolutional layers
        self.conv1 = nn.Conv2d(self.obs_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(self._feature_size(), 1024)
        self.fc2 = nn.Linear(1024, 1024)

        # LSTM layer
        self.lstm = nn.LSTM(1024, 1024, batch_first=True)

        # Actor and critic heads
        self.actor_head = nn.Linear(1024, num_outputs)
        self.critic_head = nn.Linear(1024, 1)

        self._features = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Device: {self.device}")

        # Move the model to the device
        self.to(self.device)

    def _feature_size(self) -> int:
        """
        Calculate the output size of the convolutional layers.

        Returns:
            int: The output size of the convolutional layers.
        """
        x = torch.zeros(1, *self.obs_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(np.prod(x.size()))

    def forward(self, input_dict: Dict, state: List[torch.Tensor], seq_lens: List[int]) -> Tuple[
        torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the network.

        Args:
            input_dict (Dict): The input data as a dictionary.
            state (List[torch.Tensor]): The previous hidden states of the LSTM layer.
            seq_lens (List[int]): The sequence lengths for variable-length sequences.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: The output of the network and the new hidden states.
        """
        x = input_dict["obs"].float().to(self.device)  # Convert input to float
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # LSTM processing
        batch_size = x.size(0)
        if state is None or len(state) < 2 or state[0] is None or state[1] is None:
            hx = torch.zeros(1, batch_size, 1024, device=self.device)
            cx = torch.zeros(1, batch_size, 1024, device=self.device)
        else:
            hx, cx = state
            hx = hx.to(self.device)
            cx = cx.to(self.device)
            if len(hx.shape) != 3:
                hx = hx.view(1, batch_size, 1024).contiguous()
            if len(cx.shape) != 3:
                cx = cx.view(1, batch_size, 1024).contiguous()

        x, (hx, cx) = self.lstm(x.unsqueeze(1), (hx, cx))  # Adjust sequence dimension

        self._features = x.squeeze(1)
        logits = self.actor_head(self._features.to(self.device))
        value = self.critic_head(self._features.to(self.device))

        return logits, [hx.squeeze(0), cx.squeeze(0)]

    def value_function(self) -> torch.Tensor:
        """
        Get the output of the critic head.

        Returns:
            torch.Tensor: The output of the critic head.
        """
        assert self._features is not None, "must call forward() first"
        return self.critic_head(self._features).squeeze(1)

    def initial_state(self, batch_size: int) -> tuple[Tensor, ...]:
        """
        Get the initial hidden states for the LSTM layer.

        Args:
            batch_size (int): The batch size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The initial hidden states for the LSTM layer.
        """
        return tuple(torch.zeros(self.core.num_layers, batch_size,
                                 self.core.hidden_size) for _ in range(2))


class NatureCNN(TorchModelV2, nn.Module):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    Adjusted to use the GPU if available by checking for CUDA.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # Initialize parents
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get the input channels from observation space
        # Assuming observations are in [C, H, W] format
        n_input_channels = obs_space.shape[0]

        # Define the CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),  # Adjusted kernel_size
            nn.ReLU(),
        )

        # Compute the output size after the CNN layers
        with torch.no_grad():
            sample_input = torch.zeros(
                1, n_input_channels, obs_space.shape[1], obs_space.shape[2]
            ).to(self.device)
            cnn_output = self.cnn(sample_input)
            n_flatten = cnn_output.view(1, -1).shape[1]

        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs),
        )

        # Value function head for the critic
        self.value_head = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # Move the entire model to the device
        self.to(self.device)

    def forward(self, input_dict, state, seq_lens):
        # Get observations and move to device
        obs = input_dict["obs"].float().to(self.device)
        # If observations are in [B, H, W, C], permute to [B, C, H, W]
        if obs.shape[1:] != self.obs_space.shape:
            obs = obs.permute(0, 3, 1, 2)
        # Pass through CNN
        x = self.cnn(obs)
        x = x.view(x.size(0), -1)  # Flatten

        # Store features for value function
        self._features = x

        # Pass through fully connected layers
        logits = self.fc(x.to(self.device))
        return logits, state

    def value_function(self):
        return self.value_head(self._features).squeeze(1)
