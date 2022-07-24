#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/23 21:47
# @Author  : Ly
# @File    : test_conv_function.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
from unittest import TestCase
from model.ResNet.blocks.conv_function import *


class Test(TestCase):
    def test_conv3x3(self):
        conv1 = conv3x3(10, 30, 2)
        print(conv1)

    def test_conv1x1(self):
        conv1 = conv1x1(10, 30, 2)
        print(conv1)
