import torch
import torch.nn as nn


class BasicFullyConnected(nn.Module):
    def __init__(self, input_size, output_size):
        super(BasicFullyConnected, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class RegressionModel(nn.Module):

    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.net = nn.Sequential(
            BasicFullyConnected(input_size, 256),
            BasicFullyConnected(256, 128),
            BasicFullyConnected(128, 64),
            BasicFullyConnected(64, 32),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)