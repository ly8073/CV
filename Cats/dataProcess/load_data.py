#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/5/5 21:36
# @Author  : Ly
# @File    : load_data.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
import os

from torch.utils.data import Dataset
from PIL import Image

from utils.functions import onehot_label


class CatsDataSet(Dataset):
    def __init__(self, img_dir, trans_form=None, is_train=True):
        super(CatsDataSet, self).__init__()
        self.root_dir = rf"D:\02 Coding\05 DataBase\02 Cats"
        self.img_dir = os.path.join(self.root_dir, img_dir)
        self.img_lists = os.listdir(self.img_dir)
        if is_train:
            self.label = self._load_label()
        self.trans_form = trans_form

    def _load_label(self):
        label_file = os.path.join(self.root_dir, 'train_list.txt')
        labels = {}
        with open(label_file, 'r') as f:
            for i in range(len(self.img_lists)):
                line = f.readline()
                img_name, label = line.split("\t")
                labels[img_name.split("/")[-1]] = (onehot_label(int(label), numclass=12))
        return labels

    def __getitem__(self, index):
        img_name = self.img_lists[index]
        img_file = os.path.join(self.img_dir, img_name)
        print(img_name)
        image = Image.open(img_file).convert("RGB")
        if self.trans_form is not None:
            image = self.trans_form(image)
        return image, self.label[img_name]

    def __len__(self):
        return len(self.img_lists)
