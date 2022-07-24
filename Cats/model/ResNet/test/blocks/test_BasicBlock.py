#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/23 21:51
# @Author  : Ly
# @File    : test_BasicBlock.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
from unittest import TestCase

import torch

from model.ResNet.blocks.BasicBlock import BasicBlock


class TestBasicBlock(TestCase):
    def test_forward(self):
        basic_block = BasicBlock(10, 40)
        x = torch.randn(1, 10, 28, 28)
        y = basic_block(x)
        print(y.shape)
