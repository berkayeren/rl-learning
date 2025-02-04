from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
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
        ).to(self.device)

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
        ).to(self.device)

        # Value function head for the critic
        self.value_head = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        ).to(self.device)

        # Ensure that the entire model is on the device
        self.to(self.device)

    @override(TorchModelV2)
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

        # Pass through value head to get state value
        # value = self.value_head(x).squeeze(-1)  # Remove last dimension

        return logits, state

    def value_function(self):
        return self.value_head(self._features).squeeze(1)


class CustomMiniGridLSTM(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA device")
        # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        #     print("Using MPS device")
        else:
            self.device = torch.device("cpu")
            print("Using CPU device")

        # Get activation function from model_config
        activation_fn_name = model_config['custom_model_config'].get("custom_activation", "relu").lower()
        self.activation_fn = self._get_activation_function(activation_fn_name)
        # Core dimensions
        self.obs_size = int(np.prod(obs_space.shape))
        hidden_size = 256  # Reduced size for flattened input

        # Feature extraction layers
        self.network = nn.Sequential(
            nn.Linear(self.obs_size, hidden_size),
            self.activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            self.activation_fn(),
            nn.Linear(hidden_size, self.obs_size),
            self.activation_fn(),
        ).to(self.device)

        # Value branch
        self.value_branch = nn.Sequential(
            nn.Linear(self.obs_size, hidden_size),
            self.activation_fn(),
            nn.Linear(hidden_size, 1)
        ).to(self.device)

        self._initialize_weights()

        self._features = None
        self._cur_value = None

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def _get_activation_function(self, name):
        """
        Return the corresponding activation function class based on its name.
        """
        activation_map = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "softplus": nn.Softplus
        }
        return activation_map.get(name, nn.ReLU)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Move input to appropriate device
        x = input_dict["obs"].float().to(self.device)

        # Forward pass
        self._features = self.network(x)
        self._cur_value = self.value_branch(self._features).squeeze(1)

        # Move output back to CPU if needed for RLlib
        if self.device != torch.device("cpu"):
            self._features = self._features.cpu()
            self._cur_value = self._cur_value.cpu()

        return self._features, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def to_device(self, device):
        """Helper method to move model to a different device"""
        self.device = device
        self.network = self.network.to(device)
        self.value_branch = self.value_branch.to(device)
        return self


class SimpleGridModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA device")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("cpu")
            print("Using MPS device")
        else:
            self.device = torch.device("cpu")
            print("Using CPU device")

        self.obs_size = obs_space.shape[0]

        # Get activation function from model_config
        activation_fn_name = model_config['custom_model_config'].get("custom_activation", "relu").lower()
        activation_fn_value_name = model_config['custom_model_config'].get("activation_fn_value_name", "relu").lower()
        self.activation_fn = self._get_activation_function(activation_fn_name)
        self.activation_fn_value = self._get_activation_function(activation_fn_value_name)
        print(
            f"Using activation function: {activation_fn_name},Using value activation function: {activation_fn_value_name}")

        # Feature extraction layers
        self.network = nn.Sequential(
            nn.Linear(self.obs_size, 256),  # Input layer
            nn.LayerNorm(256),  # Normalization
            self.activation_fn(),
            nn.Dropout(0.2),  # Dropout for regularization

            nn.Linear(256, 128),  # Hidden layer 1
            nn.LayerNorm(128),  # Normalization
            self.activation_fn(),
            nn.Dropout(0.2),  # Dropout for regularization

            nn.Linear(128, 64),  # Hidden layer 2
            nn.LayerNorm(64),  # Normalization
            self.activation_fn(),
        ).to(self.device)

        # Output layer for action logits
        self.fc_out = nn.Linear(128, num_outputs).to(self.device)

        # Value branch
        self.value_branch = nn.Sequential(
            nn.Linear(64, 64),  # Intermediate layer
            nn.LayerNorm(64),  # Normalization
            self.activation_fn_value(),
            nn.Dropout(0.2),  # Dropout for regularization

            nn.Linear(64, 32),  # Intermediate layer
            nn.LayerNorm(32),  # Normalization
            self.activation_fn_value(),

            nn.Linear(32, 1)  # Single output for the value function
        ).to(self.device)

    def _get_activation_function(self, name):
        """
        Return the corresponding activation function class based on its name.
        """
        activation_map = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "softplus": nn.Softplus
        }
        return activation_map.get(name, nn.ReLU)

    def forward(self, input_dict, state, seq_lens):
        # Extract components from the OrderedDict
        direction = input_dict["obs"]["direction"].float().to(self.device)  # Process direction
        image = input_dict["obs"]["image"].float().to(self.device)  # Process image
        position = input_dict["obs"]["position"].float().to(self.device)  # Process position

        # Combine processed components
        combined_features = torch.cat([direction, image, position], dim=-1)

        # Pass through the network
        x = self.network(combined_features).to(self.device)
        logits = self.fc_out(x).to(self.device)

        # Save features for value computation
        self._features = x
        return logits, state

    def value_function(self):
        return self.value_branch(self._features).squeeze(-1)


class WallAwareGridModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Activation functions
        activation_fn = model_config['custom_model_config'].get("custom_activation", "relu")
        self.activation = self._get_activation(activation_fn)

        # ========== Enhanced Feature Extraction ==========
        # Branch 1: Process image (flattened grid observation)
        self.image_net = nn.Sequential(
            nn.Linear(obs_space['image'].shape[0], 256),
            nn.LayerNorm(256),
            self.activation(),
            nn.Dropout(0.2)
        )

        # Branch 2: Process position & direction (critical for wall detection)
        self.pos_dir_net = nn.Sequential(
            nn.Linear(3, 64),  # (x, y, direction)
            nn.LayerNorm(64),
            self.activation(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            self.activation()
        )

        # Merge branches
        self.merge_net = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.LayerNorm(128),
            self.activation(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            self.activation()
        )

        # Output layers
        self.action_head = nn.Linear(64, num_outputs)
        self.value_head = nn.Sequential(
            nn.Linear(64, 32),
            self.activation(),
            nn.Linear(32, 1)
        )

    def _get_activation(self, name):
        activation_map = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "softplus": nn.Softplus
        }
        return activation_map.get(name, nn.ReLU)

    def forward(self, input_dict, state, seq_lens):
        # Extract observations
        obs = input_dict["obs"]
        image = obs["image"].float().to(self.device)
        position = obs["position"].float().to(self.device)
        direction = obs["direction"].float().to(self.device)

        # Branch 1: Image features
        img_features = self.image_net(image)

        # Branch 2: Position-Direction features
        pos_dir = torch.cat([position, direction.unsqueeze(-1)], dim=-1)
        pos_dir_features = self.pos_dir_net(pos_dir)

        # Merge features
        merged = torch.cat([img_features, pos_dir_features], dim=-1)
        x = self.merge_net(merged)

        # Save for value function
        self._features = x

        return self.action_head(x), state

    def value_function(self):
        return self.value_head(self._features).squeeze(-1)


class SpatialAwareGridModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        outputs = num_outputs if num_outputs is not None else 7
        TorchModelV2.__init__(self, obs_space, action_space, outputs, model_config, name)
        nn.Module.__init__(self)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Activation function
        activation_fn = model_config['custom_model_config'].get("custom_activation", "relu")
        print(f"Using activation function: {activation_fn}")
        self.activation = self._get_activation(activation_fn)

        try:
            self.image_shape = obs_space["image"].shape  # (H, W, C)
        except TypeError:
            self.image_shape = obs_space  # (H, W, C)

        # ========== Convolutional Layers ==========
        self.conv_net = nn.Sequential(
            # Input shape: (C, H, W)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Flatten()
        )

        # Calculate conv output size dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, *self.image_shape[:2])
            conv_out_size = self.conv_net(dummy_input).shape[-1]

        # ========== Position-Direction Processing ==========
        self.pos_dir_net = nn.Sequential(
            nn.Linear(3, 64),  # (x, y, direction)
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
        )

        # ========== Merged Network ==========
        self.merge_net = nn.Sequential(
            nn.Linear(conv_out_size + 64, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024)
        )

        # Output layers
        self.action_head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, outputs),
        )

        self.value_head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Tanh(),  # Non-linearity before final output
            nn.Linear(1024, 1),
        )

    def _get_activation(self, name):
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU
        }
        return activations.get(name.lower(), nn.ReLU)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        # Process image
        image = obs["image"].float().to(self.device)

        # Ensure 4D tensor: [Batch, Channels, Height, Width]
        if image.dim() == 3:  # Single observation (add batch dim)
            image = image.unsqueeze(0)
        image = image.permute(0, 3, 1, 2)  # [B, H, W, C] → [B, C, H, W]

        img_features = self.conv_net(image)

        position = obs["position"]  # shape can vary at runtime!
        # Convert to float to avoid dtype issues
        position = position.float().to(self.device)

        if position.dim() == 2 and position.shape[1] == 2:
            # Shape: [B, 2] => the usual case: batch of (x, y)
            x = position[:, 0]
            y = position[:, 1]

        elif position.dim() == 1 and position.shape[0] == 2:
            # Shape: [2] => a single (x, y) without a batch dimension
            # We'll treat this as batch_size=1
            x = position[0].unsqueeze(0)
            y = position[1].unsqueeze(0)

        elif position.dim() == 1:
            # Shape: [B], or [1], or something meaning “there’s only one number”
            # You can decide how to handle it. If you want to treat it as x-only, then set y=0:
            x = position
            y = torch.zeros_like(position, device=self.device)

        else:
            # If you ever get here, the shape is something unexpected.
            raise ValueError(f"Unexpected shape for obs['position']: {position.shape}")

        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)

        direction = obs["direction"].float().to(self.device)

        # If shape is [B], unsqueeze to [B,1]
        if direction.dim() == 1:
            direction = direction.unsqueeze(-1)

        pos_dir = torch.cat([x, y, direction], dim=1).to(self.device)

        pos_dir_features = self.pos_dir_net(pos_dir)

        # Merge features
        merged = torch.cat([img_features, pos_dir_features], dim=1)
        x = self.merge_net(merged)
        self._features = x

        return self.action_head(x), state

    @override(TorchModelV2)
    def value_function(self):
        assert self.value_head is not None, "must call forward() first"
        return self.value_head(self._features).squeeze(-1)
