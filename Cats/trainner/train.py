#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/23 22:42
# @Author  : Ly
# @File    : train.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from config.TrainerConfigs import TrainerConfig
from model.ModelGenerater import ModelGenerater


class Trainer:
    def __init__(self, config: TrainerConfig, num_class, loss_func,
                 optimizer: Optimizer = torch.optim.Adam):
        self.config = config
        self.num_class = num_class
        self.loss_function = loss_func
        self.optimizer = optimizer
        self.model = self._get_model()

    def _get_model(self):
        return ModelGenerater.get_model(self.config.model_name, self.num_class)

    def train(self, data_sets):
        data_loader = DataLoader(data_sets, batch_size=self.config.BATCH_SIZE, shuffle=self.config.SHUFFLE)
        for i in range(self.config.EPOCH):
            for img, label in data_loader:
                y = self.model(img)
                loss = self.loss_function(label, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def eval(self, data_sets):
        data_loader = DataLoader(data_sets, batch_size=1)
        for img, label in data_loader:
            y = self.model(img)

