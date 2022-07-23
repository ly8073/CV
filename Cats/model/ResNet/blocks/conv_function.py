#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/23 20:34
# @Author  : Ly
# @File    : conv_function.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
from torch import nn as nn


def conv3x3(in_channel, out_channel, stride=1, bn=True):
    conv_ = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3),
                  stride=(stride, stride), padding=1, bias=False),
    )
    if bn:
        conv_.add_module('bn', nn.BatchNorm2d(out_channel))
    return conv_


def conv1x1(in_channel, out_channel, stride=2):
    return nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1),
                     stride=(stride, stride), bias=False)