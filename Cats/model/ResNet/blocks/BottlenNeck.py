#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/23 22:35
# @Author  : Ly
# @File    : BottlenNeck.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
import torch.nn as nn

from model.ResNet.blocks.BasicBlock import BasicBlock
from model.ResNet.blocks.conv_function import conv1x1, conv3x3


class BottlenNeck(nn.Module):
    def __init__(self, input_channel, output_channel, mid_channel, down_sample=False):
        super(BottlenNeck, self).__init__()
        self.down_sample = None
        stride = 1
        if down_sample:
            self.down_sample = conv1x1(input_channel, output_channel, 2)
            stride = 2

        self.conv1 = conv1x1(input_channel, mid_channel)
        self.conv2 = conv3x3(mid_channel, mid_channel, stride)
        self.conv3 = conv1x1(mid_channel, output_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(self.relu(out))
        out = self.conv3(self.relu(out))
        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        return self.relu(out)
