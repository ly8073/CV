#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/4/9 13:47
# @Author  : Ly
# @File    : main.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
import os.path

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms

from GAN.models.train import Trainer
from MINIST.DataPreprocess.DealData import DataProcess
import MINIST.configs as cfg


def main():
    train_set = DataProcess(cfg.FOLDER, cfg.TRAIN_IMAGES, cfg.TRAIN_LABELS,
                            transform=transforms.ToTensor())
    test_set = DataProcess(cfg.FOLDER, cfg.TEST_IMAGES, cfg.TEST_LABELS,
                           transform=transforms.ToTensor())
    trainer = Trainer()
    trainer.train(train_set, test_set)


def identify_number():
    test_set = DataProcess(cfg.FOLDER, cfg.TEST_IMAGES, cfg.TEST_LABELS, onehot=False,
                           transform=transforms.ToTensor())
    check_points = os.path.join(cfg.CHECKPOINT_FOLDER, f"checkpoints_{99}.pkl")
    dis_criminator_checkpoints = os.path.join(cfg.CHECKPOINT_FOLDER, f"discriminator_{99}.pkl")
    nets = torch.load(check_points).eval().cpu()
    dis_criminator = torch.load(dis_criminator_checkpoints).cpu()
    total, correct = 0, 0
    while True:
        try:
            test_img_number = int(input("input a number(0~999):"))
            image, label = test_set[test_img_number]
            y, fake_image = nets(image)
            dis_out = dis_criminator(fake_image)
            print(f"according to discriminator, the fake image is {dis_out} ")
            prop, target = torch.max(y, dim=0)
            print(f"label={label}\n"
                  f"judge={target.item()}, props={prop.item()}")
            plt.imshow(fake_image.detach().numpy().squeeze())
            plt.show()
            total += 1
            if target.item() == label:
                correct += 1
        except Exception:
            break
    print(f"tested {total} pictures, {correct} correct, rate={100 * correct / total}%")


if __name__ == "__main__":
    # main()
    identify_number()