#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/23 22:54
# @Author  : Ly
# @File    : test_BottlenNeck.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
from unittest import TestCase

import torch

from model.ResNet.blocks.BottlenNeck import BottlenNeck


class TestBottlenNeck(TestCase):
    def test_forward(self):
        bottlen_neck = BottlenNeck(256, 128)
        x = torch.randn(1, 256, 56, 56)
        y = bottlen_neck(x)
        print(y.shape)
        print(bottlen_neck)
