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
        x = input_dict["obs"].float()  # Convert input to float
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # LSTM processing
        batch_size = x.size(0)
        if state is None or len(state) < 2 or state[0] is None or state[1] is None:
            hx = torch.zeros(1, batch_size, 1024, device=x.device)
            cx = torch.zeros(1, batch_size, 1024, device=x.device)
        else:
            hx, cx = state
            if len(hx.shape) != 3:
                hx = hx.view(1, batch_size, 1024).contiguous()
            if len(cx.shape) != 3:
                cx = cx.view(1, batch_size, 1024).contiguous()

        x, (hx, cx) = self.lstm(x.unsqueeze(1), (hx, cx))  # Adjust sequence dimension

        self._features = x.squeeze(1)
        logits = self.actor_head(self._features)
        value = self.critic_head(self._features)

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
