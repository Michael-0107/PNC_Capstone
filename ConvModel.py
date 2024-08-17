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
    def __init__(self, feature_size, num_conv_layers=2, dropout=False, batch_norm=False, activation_fn=nn.ReLU):
        super(ConvModel, self).__init__()
        layers = []

        in_channels = feature_size
        out_channels = 32

        for i in range(num_conv_layers):
            layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(activation_fn())
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            if dropout:
                layers.append(nn.Dropout(0.3))
            in_channels = out_channels
            out_channels *= 2
        
        self.convs = nn.Sequential(*layers)
        
        fc_layers = []
        fc_layers.append(nn.Linear(64, 32))
        if batch_norm:
            fc_layers.append(nn.BatchNorm1d(32))
        fc_layers.append(activation_fn())
        if dropout:
            fc_layers.append(nn.Dropout(0.3))
        
        fc_layers.append(nn.Linear(32, 16))
        if batch_norm:
            fc_layers.append(nn.BatchNorm1d(16))
        fc_layers.append(activation_fn())
        if dropout:
            fc_layers.append(nn.Dropout(0.3))
        
        fc_layers.append(nn.Linear(16, 1))
        fc_layers.append(nn.Sigmoid())
        
        self.fcs = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        x = self.fcs(x)
        return x


if __name__ == "__main__":
    model = ConvModel(24)
    random_tensor = torch.rand(16, 8, 24)

    out = model(random_tensor)

    print(out.shape)



