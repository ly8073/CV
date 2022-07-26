#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/24 17:49
# @Author  : Ly
# @File    : TrainerConfigs.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
import torch
import torchvision.transforms as transform


class TrainerConfig:
    loss_dict = {
        'cross_entropy': torch.nn.CrossEntropyLoss,
        'mse': torch.nn.MSELoss,
    }

    def __init__(self, model_name, num_class, loss_name='cross_entropy'):
        self.model_name = model_name
        self.num_class = num_class

        # during train
        self.EPOCH = 30
        self.LEARNING_RATE = 1e-3
        self.BATCH_SIZE = 24
        self.SHUFFLE = True
        self.pretrain_path = None
        self.loss_func = self._get_loss_func(loss_name)
        self.optimizer = torch.optim.Adam
        self.batch_log_num = 5
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # img_trans
        self.trans = transform.Compose([
            transform.RandomCrop(224, 224),
            transform.RandomRotation(180),
            transform.ToTensor(),
            transform.Normalize(mean=[0.4849716, 0.44362384, 0.4024527],
                                std=[0.27388313, 0.2682838, 0.27511856])
        ])

    def _get_loss_func(self, loss_name):
        return self.loss_dict.get(loss_name, torch.nn.CrossEntropyLoss)()
