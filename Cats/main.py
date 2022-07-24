#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/24 23:42
# @Author  : Ly
# @File    : main.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
from trainner.train import Trainer
from config.TrainerConfigs import TrainerConfig


def main():
    trainer_config = TrainerConfig('ResNet_18')
    trainer = Trainer(trainer_config, 30, None)
    trainer.train()
    trainer.eval()


if __name__ == "__main__":
    main()