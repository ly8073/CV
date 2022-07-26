#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/24 17:49
# @Author  : Ly
# @File    : TrainerConfigs.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
import os.path

import torch


class TrainerConfig:
    def __init__(self, model_name, pretrain_path=None):
        self.model_name = model_name
        self.EPOCH = 30
        self.LEARNING_RATE = 1e-3
        self.BATCH_SIZE = 24
        self.SHUFFLE = True
        if os.path.exists(pretrain_path):
            self.pretrain_dict = torch.load(pretrain_path)



