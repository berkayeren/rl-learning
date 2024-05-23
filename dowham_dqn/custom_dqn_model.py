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

        self.obs_shape = obs_space.shape
        self.num_actions = action_space.n

        # Adjusted to handle image observations from ImgObsWrapper
        self.conv1 = nn.Conv2d(self.obs_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(self._feature_size(), 1024)
        self.fc2 = nn.Linear(1024, 1024)

        self.lstm = nn.LSTM(1024, 1024, batch_first=True)

        self.actor_head = nn.Linear(1024, num_outputs)
        self.critic_head = nn.Linear(1024, 1)

        self._features = None

    def _feature_size(self):
        x = torch.zeros(1, *self.obs_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(np.prod(x.size()))

    def forward(self, input_dict, state, seq_lens):

        if state is None or len(state) == 0:
            state = [torch.zeros(1, 32, 1024), torch.zeros(1, 32, 1024)]

        x = input_dict["obs"].float()  # Convert input to float
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Debugging: Print state information
        print(f"State before processing: {state}")
        print(f"State length: {len(state)}")

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

        # Debugging: Print state information after processing
        print(f"State after processing: {[hx.squeeze(0), cx.squeeze(0)]}")

        return logits, [hx.squeeze(0), cx.squeeze(0)]

    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return self.critic_head(self._features).squeeze(1)

    def get_initial_state(self):
        # Return initial hidden states for LSTM
        h = [torch.zeros(1, 1, 1024), torch.zeros(1, 1, 1024)]
        return h
