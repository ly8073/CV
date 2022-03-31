import torch
import torch.nn as nn
from torch.nn import Module


class NetWork(Module):
    def __init__(self, input_channel, out_put):
        super(NetWork, self).__init__()
        self.input_channel = input_channel
        self.out_put = out_put
        self.convs = nn.Sequential(
            nn.Conv2d(self.input_channel, 32, 3, 1),
            nn.Conv2d(32, 64, 3, 1),
        )
        self.fcs = nn.Sequential(
            nn.Linear(9168, 128),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        y = self.convs(x)
        # props = self.fcs(y)
        return y


net = NetWork(1, 10)
print(net)