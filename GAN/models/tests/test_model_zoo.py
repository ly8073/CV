#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 21:38
# @Author  : Ly
# @File    : test_model_zoo.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
from unittest import TestCase

import torch
from GAN.models.model_zoo import DecodeFromNum, GanForMinist, Discriminator
from GAN.models.train import generate_random_number


class TestModelZoo(TestCase):
    def test_decode_forward(self):
        net = DecodeFromNum((32, 7, 7), 1568, 16)
        x = generate_random_number(10)
        y = net(x)
        print(y.shape)

    def test_gan_forward(self):
        net = GanForMinist(1, 16, 32)
        x = torch.randn(1, 28, 28)
        y = net(x)
        print(y.shape)

    def test_discriminator_forward(self):
        net = Discriminator(28 * 28, 1)
        x = torch.randn(1, 28, 28)
        y = net(x)
        print(y.shape)
