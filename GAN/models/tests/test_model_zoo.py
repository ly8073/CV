#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 21:38
# @Author  : Ly
# @File    : test_model_zoo.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
import torch
from GAN.models.model_zoo import DecodeFromNum, GanForMinist


def test_decode_forward():
    net = DecodeFromNum((32, 7, 7), 1568, 16)
    x = torch.randn(10)
    y = net(x)
    print(y.shape)


def test_gan_forward():
    net = GanForMinist(1, 16, 32)
    x = torch.randn(1, 28, 28)
    y = net(x)
    print(y.shape)
