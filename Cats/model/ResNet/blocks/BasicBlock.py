#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/23 19:57
# @Author  : Ly
# @File    : BasicBlock.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073

import torch.nn as nn

from Cats.model.ResNet.blocks.conv_function import conv3x3, conv1x1


class BasicBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(BasicBlock, self).__init__()
        self.output_channel = output_channel
        self.down_sample = None
        stride = 1
        if input_channel != self.output_channel:
            self.down_sample = conv1x1(input_channel, self.output_channel, 2)
            stride = 2

        self.conv1 = conv3x3(input_channel, self.output_channel, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(self.output_channel, self.output_channel)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)
        return out
