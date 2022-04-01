import torch
import torch.nn as nn
from torch.nn import Module


class NetWork(Module):
    def __init__(self, input_channel, hidden_channels):
        super(NetWork, self).__init__()
        self.input_channel = input_channel
        self.hidden_channels = hidden_channels
        self.convs = nn.Sequential(
            nn.Conv2d(self.input_channel, hidden_channels, 3, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 32, 3, 1),
            nn.ReLU(),
        )
        self.fcs = nn.Sequential(
            nn.Linear(18432, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=0),
        )

    def forward(self, x):
        y = self.convs(x)
        y = y.reshape(y.shape[0], -1)
        props = self.fcs(y)
        return props

#
# net = NetWork(1, 16)
# x = torch.rand(10, 1, 28, 28)
# y = net(x)
# print("done")