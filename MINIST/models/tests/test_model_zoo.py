#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 21:07
# @Author  : Ly
# @File    : test_model_zoo.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
import torch

from MINIST.models.model_zoo import NetWork


def test_forward():
    nets = NetWork(1, 16, 32)
    x = torch.randn(1, 28, 28)
    y = nets(x)
    print(nets.flatten_dim)
