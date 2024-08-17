from sympy import Basic
import torch
import torch.nn as nn

from Hypers import Config
import Hypers


class BasicConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(BasicConvBlock, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=out_channels),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        # x.shape = (B, in_channels, L)
        x = self.nn(x) # (B, out_channels, L/2)
        return x


class ConvModel(nn.Module):
    def __init__(self, feature_size, sequence_length):
        super(ConvModel, self).__init__()
        self.convs = nn.Sequential(
            BasicConvBlock(in_channels=feature_size, out_channels=32),
            BasicConvBlock(in_channels=32, out_channels=64)
        )

        self.fc_input_size = 64 * (sequence_length // 4)
        self.fcs = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, L)
        x = self.convs(x)  # (B, 64, L/4)
        x = x.view(x.shape[0], -1)  # (B, 64*L/4)
        x = self.fcs(x)  # (B, 1)
        return x

class ConvModelV2(nn.Module):
    def __init__(self, feature_size, sequence_length=4):
        super(ConvModelV2, self).__init__()
        self.convs = nn.Sequential(
            BasicConvBlock(in_channels=feature_size, out_channels=32)
        )

        self.fc_input_size = 32 * (sequence_length // 2)
        self.fcs = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, L)
        x = self.convs(x)  # (B, 32, L/2)
        x = x.view(x.shape[0], -1)  # (B, 32*L/2)
        x = self.fcs(x)  # (B, 1)
        return x


if __name__ == "__main__":
    model = ConvModel(24)
    random_tensor = torch.rand(16, 8, 24)

    out = model(random_tensor)

    print(out.shape)



