import torch
import torch.nn as nn
from torch.nn import Module


class NetWork(Module):
    def __init__(self, input_channel, hidden_channels, output_channel):
        super(NetWork, self).__init__()
        self.input_channel = input_channel
        self.hidden_channels = hidden_channels
        self.convs = nn.Sequential(
            nn.Conv2d(self.input_channel, hidden_channels, (3, 3), (2, 2), 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, output_channel, (3, 3), (2, 2), 1),
            nn.ReLU(),
        )
        self._calculate_fc_dims()
        self.fcs = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=0),
        )

    def _calculate_fc_dims(self):
        x = torch.randn(1, 28, 28)
        y = self.convs(x)
        self.img_shape = y.shape
        self.flatten_dim = y.flatten().shape[0]

    def forward(self, x):
        y = self.convs(x)
        if len(y.shape) == 4:
            y = y.reshape(y.shape[0], -1)
        else:
            y = y.reshape(-1)
        props = self.fcs(y)
        return props
