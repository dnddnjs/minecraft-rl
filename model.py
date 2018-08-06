import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import HyperParams as hp
from torchvision import transforms


class Actor(nn.Module):
    def __init__(self, num_outputs):
        self.num_outputs = num_outputs
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4,
                               out_channels=16,
                               kernel_size=8,
                               stride=4)

        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=4,
                               stride=2)

        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               stride=1)

        self.fc = nn.Linear(7*7*64, num_outputs)
        self.fc.weight.data.mul_(0.1)
        self.fc.bias.data.mul_(0.0)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        mu = self.fc(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, std, logstd


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4,
                               out_channels=16,
                               kernel_size=8,
                               stride=4)

        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=4,
                               stride=2)

        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               stride=1)

        self.fc = nn.Linear(7*7*64, 1)
        self.fc.weight.data.mul_(0.1)
        self.fc.bias.data.mul_(0.0)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        v = self.fc(x)
        return v
