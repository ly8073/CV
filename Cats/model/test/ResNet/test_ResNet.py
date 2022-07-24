#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/24 15:59
# @Author  : Ly
# @File    : test_ResNet.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
from unittest import TestCase

import torch

from config.ResNetConfigs import *
from model.ResNet.ResNet import ResNet
from config.ResNetConfigs import get_resnet


class TestResNet(TestCase):
    def test_forward(self):
        configs = ResNet50Config()
        net = ResNet(configs.block, configs.repeat_time, configs.mid_channels)
        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        print(y.shape)
        print(net)

    def test_get_resnet(self):
        net = get_resnet(ResNet18Config(), 30)
        print(net)
