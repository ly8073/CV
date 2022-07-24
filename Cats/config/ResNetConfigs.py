#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/24 16:08
# @Author  : Ly
# @File    : ResNetConfigs.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
from config.ModelConfig import ModelConfig
from model.ResNet.ResNet import ResNet
from model.ResNet.blocks import *


class ResNetConfig(ModelConfig):
    def __init__(self):
        super(ResNetConfig, self).__init__()
        self.block = None
        self.repeat_time = None
        self.mid_channels = [64, 128, 256, 512]


def get_resnet(config: ResNetConfig, num_classes=1000):
    net = ResNet(config.block, config.repeat_time, config.mid_channels, num_classes)
    return net


class ResNet18Config(ResNetConfig):
    def __init__(self):
        super(ResNet18Config, self).__init__()
        self.block = BasicBlock.BasicBlock
        self.repeat_time = [2, 2, 2, 2]


class ResNet34Config(ResNetConfig):
    def __init__(self):
        super(ResNet34Config, self).__init__()
        self.block = BasicBlock.BasicBlock
        self.repeat_time: list = [3, 4, 6, 3]


class ResNet50Config(ResNetConfig):
    def __init__(self):
        super(ResNet50Config, self).__init__()
        self.block = BottlenNeck.BottlenNeck
        self.repeat_time: list = [3, 4, 6, 3]


class ResNet101Config(ResNetConfig):
    def __init__(self):
        super(ResNet101Config, self).__init__()
        self.block = BottlenNeck.BottlenNeck
        self.repeat_time: list = [3, 4, 23, 3]


class ResNet152Config(ResNetConfig):
    def __init__(self):
        super(ResNet152Config, self).__init__()
        self.block = BottlenNeck.BottlenNeck
        self.repeat_time: list = [3, 8, 36, 3]


resnet_dict = {
    'gene': get_resnet,
    '18': ResNet18Config,
    '34': ResNet34Config,
    '50': ResNet50Config,
    '101': ResNet101Config,
    '152': ResNet152Config,

}
