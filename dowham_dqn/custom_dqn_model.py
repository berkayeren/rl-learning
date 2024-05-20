import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class CustomDQNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        conv_out_size = self._get_conv_output_size(obs_space.shape)
        self.fc1 = nn.Linear(conv_out_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        self.lstm = nn.LSTM(1024, 1024, batch_first=True)

        self.actor_head = nn.Linear(1024, num_outputs)
        self.critic_head = nn.Linear(1024, 1)

    def _get_conv_output_size(self, shape):
        o = torch.zeros(1, *shape)
        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        return int(np.prod(o.size()))

    def forward(self, input_dict, state, seq_lens):
        image = input_dict["obs"]["image"]
        direction = input_dict["obs"]["direction"].view(-1, 1)
        mission = input_dict["obs"]["mission"].view(-1, 1)

        x = F.relu(self.conv1(image.permute(0, 3, 1, 2)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = torch.cat([x, direction, mission], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x, state = self.lstm(x.unsqueeze(1), state)
        x = x.squeeze(1)

        actor_logits = self.actor_head(x)
        critic_value = self.critic_head(x)

        return actor_logits, state

    def value_function(self):
        return self.critic_head

    def get_initial_state(self):
        h, c = [torch.zeros(1, self.lstm.hidden_size)] * 2
        return [h, c]
