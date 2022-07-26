#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/23 22:42
# @Author  : Ly
# @File    : train.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
import logging
import os.path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config.TrainerConfigs import TrainerConfig
from model.ModelGenerater import Generater


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.loss_function = self.config.loss_func

        self.logger = self._get_logger()
        self.logger.setLevel('INFO')
        self.model = self._get_model().to(config.device)
        self.optimizer = config.optimizer(self.model.parameters(), lr=config.LEARNING_RATE)

        self.summary = SummaryWriter("./logs")

    def _get_logger(self):
        logging.basicConfig(level=logging.INFO, filemode='a',
                            format='【%(asctime)s】 【%(levelname)s】 【%(name)s】>>>  %(message)s', datefmt='%Y-%m-%d %H:%M')
        logger = logging.getLogger(__name__)
        return logger

    def _get_model(self):
        net = Generater.get_model(self.config.model_name, self.config.num_class)
        self.logger.info(f'get model {self.config.model_name} done....')
        if self.config.pretrain_path is not None and os.path.exists(self.config.pretrain_path):
            model_dict = torch.load(self.config.pretrain_path)
            net.load_state_dict(model_dict)
            self.logger.info('load pretrain dict done....')
        return net

    def train(self, data_sets: torch.utils.data.Dataset):
        data_loader = DataLoader(data_sets, batch_size=self.config.BATCH_SIZE, shuffle=self.config.SHUFFLE)
        batch_log_period = len(data_loader) // self.config.batch_log_num + 1
        batch_num = 0
        for i in range(self.config.EPOCH):
            self.logger.info(f'begin to train epoch [{i}/{self.config.EPOCH}]:')
            for j, (img, label) in enumerate(data_loader):
                img, label = img.to(self.config.device), label.to(self.config.device)
                y = self.model(img)
                loss = self.loss_function(label, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if (j + 1) % batch_log_period == 0:
                    self.logger.info(f'\tbatch [{j + 1} / {len(data_loader)}]: loss = {loss.item()}')

                self.summary.add_scalar('train_loss', loss.item(), batch_num)
                batch_num += 1

    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def eval(self, data_sets):
        data_loader = DataLoader(data_sets, batch_size=1)
        for img, label in data_loader:
            y = self.model(img)

