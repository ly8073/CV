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

    def forward(self, x):
        numbers = self.encode(x)
        fake_imgs = self.decode(numbers)
        return numbers, fake_imgs


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
