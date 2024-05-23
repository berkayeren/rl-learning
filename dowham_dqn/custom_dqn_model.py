import numpy as np
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class CustomDQNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Calculate the output size of the convolutional layers
        conv_out_size = self._get_conv_output_size((15, 15, 3))  # Use the correct shape for the image

        # Define the fully connected layers
        self.fc1 = nn.Linear(conv_out_size + 1 + 1, 1024)  # Include direction and mission sizes
        self.fc2 = nn.Linear(1024, 1024)

        # Define the LSTM layer
        self.lstm = nn.LSTM(1024, 1024, batch_first=True)

        # Define the output layers for the actor and critic
        self.actor_head = nn.Linear(1024, num_outputs)
        self.critic_head = nn.Linear(1024, 1)

    def _get_conv_output_size(self, shape):
        if len(shape) != 3:
            raise ValueError(f"Expected shape with 3 dimensions, got {shape}")

        # Create a dummy input to calculate the output size after convolutions
        o = torch.zeros(1, shape[2], shape[0], shape[1])
        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        return int(np.prod(o.size()))

    def forward(self, input_dict, state, seq_lens):
        # Process the image input
        image = input_dict["obs"]["image"]
        x = F.relu(self.conv1(image.permute(0, 3, 1,
                                            2)))  # [batch_size, height, width, channels] -> [batch_size, channels, height, width]
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the convolutional output
        x = x.reshape(x.size(0), -1)

        # Process direction and mission inputs
        direction = input_dict["obs"]["direction"].view(x.size(0), -1)
        mission = input_dict["obs"]["mission"].view(x.size(0), -1)

        # Concatenate all inputs
        x = torch.cat([x, direction, mission], dim=1)

        # Assuming x is your input tensor and self.fc1 is the linear layer
        x = x.view(x.size(0), -1)  # Flatten the tensor
        if x.size(1) != self.fc1.weight.size(1):
            x = F.pad(x, (0, self.fc1.weight.size(1) - x.size(1)))  # Pad the tensor if necessary

        x = F.relu(self.fc1(x))  # Now you can pass the tensor through the linear layer
        x = F.relu(self.fc2(x))

        # Initialize the hidden state and cell state to zeros if they are not provided
        if len(state) == 0:
            # Add the batch size to the dimensions of h and c
            h = torch.zeros(x.size(0), 1, self.lstm.hidden_size)
            c = torch.zeros(x.size(0), 1, self.lstm.hidden_size)
        else:
            h, c = state

        # Transpose the dimensions of the hidden state tensor
        h = h.transpose(0, 1)
        c = c.transpose(0, 1)

        # Pass the input tensor and the initial state to the LSTM layer
        x, (h, c) = self.lstm(x.unsqueeze(1), (h, c))

        # The LSTM layer returns an output tensor and a new state
        # The new state is a tuple of two tensors: the hidden state and the cell state
        # Convert the state to a list before returning it
        state = [h, c]

        # Output layers for actor and critic
        actor_logits = self.actor_head(x)
        critic_value = self.critic_head(x)

        return actor_logits, state

    def value_function(self):
        return self.critic_head

    def get_initial_state(self):
        h, c = [torch.zeros(1, self.lstm.hidden_size)] * 2
        return [h, c]


from torch import nn
import torch
import torch.nn.functional as F


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class CustomMinigridPolicyNet(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Assuming the observation space is a Dict with 'image', 'direction', and 'mission'
        self.observation_shape = obs_space.shape
        self.num_actions = action_space.n

        init_ = lambda m: init(m, nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )

        self.fc = nn.Sequential(
            init_(nn.Linear(32 * ((self.observation_shape[0] // 8) ** 2), 1024)),
            # Adjust the input size to match the output of convolutional layers
            nn.ReLU(),
            init_(nn.Linear(1024, 1024)),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(1024, 1024, batch_first=True)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.actor_head = nn.Sequential(
            init_(nn.Linear(1024, 1024)),
            nn.ReLU(),
            init_(nn.Linear(1024, self.num_outputs))
        )
        self.critic_head = nn.Sequential(
            init_(nn.Linear(1024, 1024)),
            nn.ReLU(),
            init_(nn.Linear(1024, 1))
        )

    def initial_state(self, batch_size):
        return tuple(torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size) for _ in range(2))

    def forward(self, input_dict, state, seq_lens):
        image = input_dict['obs']['image']
        direction = input_dict['obs']['direction']
        mission = input_dict['obs']['mission']

        # Process image
        T, B, C, H, W = image.shape  # Assuming the image tensor has dimensions (batch_size, channels, height, width)
        x = image.view(T * B, C, H, W).float()  # Flatten the batch and sequence dimensions

        x = self.feat_extract(x)
        x = x.view(T * B, -1)
        x = self.fc(x)

        x = x.view(T, B, -1)
        x, state = self.lstm(x, state)
        x = x.contiguous().view(T * B, -1)

        actor_logits = self.actor_head(x)
        critic_value = self.critic_head(x)

        if self.training:
            action = torch.multinomial(F.softmax(actor_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(actor_logits, dim=1)

        actor_logits = actor_logits.view(T, B, self.num_actions)
        critic_value = critic_value.view(T, B)
        action = action.view(T, B)

        return dict(policy_logits=actor_logits, baseline=critic_value, action=action), state

    def value_function(self):
        return self.critic_head

    def get_initial_state(self):
        h, c = [torch.zeros(1, self.lstm.hidden_size)] * 2
        return [h, c]
