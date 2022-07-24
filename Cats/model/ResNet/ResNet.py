#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/24 0:05
# @Author  : Ly
# @File    : ResNet.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
from typing import Union, Type

import torch
from torch import nn

from model.ResNet.blocks.BasicBlock import BasicBlock
from model.ResNet.blocks.BottlenNeck import BottlenNeck
from model.ResNet.blocks.conv_function import conv7x7


class ResNet(nn.Module):
    def __init__(self, block, repeat_time_list, mid_channel_list, num_class=1000):
        super(ResNet, self).__init__()
        self.input_channel = 64

        self.conv1 = conv7x7(3, self.input_channel, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)

        self.nets = nn.Sequential()
        for i, (repeat, mid_channel) in enumerate(zip(repeat_time_list, mid_channel_list)):
            layer = self._make_layers(block, mid_channel, repeat)
            self.nets.add_module(f'conv_{i + 2}', layer)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.out_channel, num_class)
        self.soft_max = nn.Softmax(dim=1)

    def _make_layers(self, block: Type[Union[BasicBlock, BottlenNeck]], mid_channel, repeat):
        layers = [block(self.input_channel, mid_channel)]
        self.input_channel = layers[0].output_channel
        for _ in range(1, repeat):
            tmp = block(self.input_channel, mid_channel)
            self.out_channel = tmp.output_channel
            layers.append(tmp)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.nets(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        y = self.soft_max(x)
        return y
