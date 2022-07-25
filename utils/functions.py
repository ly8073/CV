#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/25 21:58
# @Author  : Ly
# @File    : functions.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
import torch


def onehot_label(target, numclass=10):
    onehot = [0] * numclass
    onehot[target] = 1
    return torch.Tensor(onehot)