#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 21:14
# @Author  : Ly
# @File    : model_zoo.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
import torch
import torch.nn as nn
from MINIST.models.model_zoo import NetWork


class GanForMinist(nn.Module):
    def __init__(self, in_channel, hidden_channel, img_channel):
        super(GanForMinist, self).__init__()
        self.encode = NetWork(in_channel, hidden_channel, img_channel)
        self.decode = DecodeFromNum(self.encode.img_shape, self.encode.flatten_dim, hidden_channel)

    def forward(self, data, label=None):
        numbers = self.encode(data)
        if label is None:
            fake_images = self.decode(numbers)
        else:
            fake_images = self.decode(label)
        # fake_images = self.decode(x)
        # numbers = self.encode(fake_images)
        return numbers, fake_images


class DecodeFromNum(nn.Module):
    def __init__(self, in_pic_size, flatten_dim, hidden_channel):
        super(DecodeFromNum, self).__init__()
        self.in_pic_size = in_pic_size
        self.fcs = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, flatten_dim),
            nn.ReLU()
        )
        self.anti_convs = nn.Sequential(
            nn.ConvTranspose2d(in_pic_size[0], hidden_channel, (3, 3), (2, 2), padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channel, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        imgs = self.fcs(x)
        if len(imgs.shape) == 2:
            imgs = imgs.reshape((imgs.shape[0],) + self.in_pic_size)
        else:
            imgs = imgs.reshape(self.in_pic_size)
        fake_imgs = self.anti_convs(imgs)
        return fake_imgs


class Discriminator(nn.Module):
    def __init__(self, indim, out_dim=1):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(indim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.flatten()
        else:
            x = x.reshape(x.shape[0], -1)
        y = self.fc(x)
        return y