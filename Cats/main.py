#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/24 23:42
# @Author  : Ly
# @File    : main.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
from dataProcess.load_data import CatsDataSet
from trainner.train import Trainer
from config.TrainerConfigs import TrainerConfig


def main(model_name, num_class):
    trainer_config = TrainerConfig(model_name, num_class)
    trainer = Trainer(trainer_config)

    cats_data = CatsDataSet("cat_12_train", num_class=trainer_config.num_class, trans_form=trainer_config.trans)
    trainer.train(cats_data)


if __name__ == "__main__":
    main('ResNet_18', 12)