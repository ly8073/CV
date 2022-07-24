#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/23 22:42
# @Author  : Ly
# @File    : train.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
from torchvision.models.resnet import resnet18


if __name__ == "__main__":
    net = resnet18()
    print(net)