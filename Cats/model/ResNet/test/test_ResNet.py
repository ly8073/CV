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
from model.ResNet.config.configs import *
from model.ResNet.blocks.BottlenNeck import BottlenNeck


class TestResNet(TestCase):
    def test_forward(self):
        configs = ResNet50Config
        net = ResNet(configs.block, configs.repeat_time, configs.mid_channels)
        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        print(y.shape)
        print(net)
