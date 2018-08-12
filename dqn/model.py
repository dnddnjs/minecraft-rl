import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import HyperParams as hp
from torchvision import transforms


class QNet(nn.Module):
    def __init__(self, num_outputs):
        self.num_outputs = num_outputs
        super(QNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4,
                               out_channels=16,
                               kernel_size=5,
                               stride=2)

        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=5,
                               stride=2)

        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=5,
                               stride=2)

        self.fc1 = nn.Linear(7*7*32, 512)
        self.fc2 = nn.Linear(512, num_outputs)
        self.fc2.weight.data.mul_(0.1)
        self.fc2.bias.data.mul_(0.0)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        q_value = self.fc2(x)
        return q_value
