#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/5/5 22:01
# @Author  : Ly
# @File    : test_load_data.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from Cats.dataProcess.load_data import CatsDataSet
import torchvision.transforms as transform


class Test(TestCase):
    def test__load_label(self):
        cats_data = CatsDataSet("cat_12_train")
        print(len(cats_data.label))

    def test_cats_data_set(self):
        cats_data = CatsDataSet("cat_12_train")
        image, label = cats_data[700]
        plt.imshow(image)
        plt.show()
        print(label)

    def test_mean_std(self):
        trans = transform.Compose([
            transform.Resize((300, 300)),
            transform.ToTensor(),
        ])
        train_data = CatsDataSet('cat_12_train', trans_form=trans)
        train_loader = DataLoader(dataset=train_data, batch_size=2160, shuffle=True)  # 3000张图片的mean std
        train = iter(train_loader).next()[0]  # 3000张图片的mean、std
        train_mean = np.mean(train.numpy(), axis=(0, 2, 3))
        train_std = np.std(train.numpy(), axis=(0, 2, 3))
        print(train_mean)
        print(train_std)

    def test_load_data(self):
        trans = transform.Compose([
            transform.RandomCrop(200, 200),
            transform.RandomRotation(180),
            transform.ToTensor(),
            transform.Normalize(mean=[0.4849716, 0.44362384, 0.4024527],
                                std=[0.27388313, 0.2682838, 0.27511856])
        ])
        train_data = CatsDataSet('cat_12_train', trans_form=trans)
        train_loader = DataLoader(dataset=train_data, batch_size=30, shuffle=True)
        for image, label in train_loader:
            print(image.shape)
            print(label.shape)
