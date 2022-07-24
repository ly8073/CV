#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/24 15:59
# @Author  : Ly
# @File    : test_ResNet.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
from unittest import TestCase

import torch

from model.ResNet.ResNet import ResNet
from model.ResNet.blocks.BasicBlock import BasicBlock
from model.ResNet.blocks.BottlenNeck import BottlenNeck


class TestResNet(TestCase):
    def test_forward(self):
        repeat_time = [3, 4, 6, 3]
        mid_channel = [64, 128, 256, 512]
        net = ResNet(BottlenNeck, repeat_time, mid_channel)
        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        print(y.shape)
        print(net)
