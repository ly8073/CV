#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/25 0:00
# @Author  : Ly
# @File    : test_ModelGenerater.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
from unittest import TestCase

from model.ModelGenerater import ModelGenerater


class TestModelGenerater(TestCase):
    def test_get_model(self):
        net = ModelGenerater.get_model('ResNet_18', 30)
        print(net)
