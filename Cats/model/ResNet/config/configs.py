#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/24 16:08
# @Author  : Ly
# @File    : configs.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
from model.ResNet.blocks import *


class ResNetConfig:
    mid_channels: list = [64, 128, 256, 512]


class ResNet18Config(ResNetConfig):
    block = BasicBlock.BasicBlock
    repeat_time: list = [2, 2, 2, 2]


class ResNet34Config(ResNetConfig):
    block = BasicBlock.BasicBlock
    repeat_time: list = [3, 4, 6, 3]


class ResNet50Config(ResNetConfig):
    block = BottlenNeck.BottlenNeck
    repeat_time: list = [3, 4, 6, 3]


class ResNet101Config(ResNetConfig):
    block = BottlenNeck.BottlenNeck
    repeat_time: list = [3, 4, 23, 3]


class ResNet152Config(ResNetConfig):
    block = BottlenNeck.BottlenNeck
    repeat_time: list = [3, 8, 36, 3]
