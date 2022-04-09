#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/4/9 13:33
# @Author  : Ly
# @File    : train.py
# @Software: PyCharm
# @Github  : https://github.com/ly8073
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from GAN.models.model_zoo import GanForMinist
import MINIST.configs as cfg


def train(train_set, test_set):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    nets = GanForMinist(1, 16, 32).to(device)
    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE)
    optimizer = torch.optim.Adam(nets.parameters(), lr=cfg.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    train_losses, test_losses = [], []
    for i in range(cfg.EPOCH):
        loss = train_one_epoch(train_loader, nets, device, criterion, optimizer, i)
        train_losses.append(loss)
        eval_loss, correct_rate = eval_one_epoch(test_loader, nets, device, criterion)
        test_losses.append(eval_loss)
        if i % cfg.CHECKPOINT == (cfg.CHECKPOINT - 1):
            if not os.path.exists(cfg.CHECKPOINT_FOLDER):
                os.mkdir(cfg.CHECKPOINT_FOLDER)
            torch.save(nets, os.path.join(cfg.CHECKPOINT_FOLDER, f"checkpoints_{i}.pkl"))
        print(f"epoch [{i + 1} / {cfg.EPOCH}] trained done. "
              f"\n\tloss={loss}, "
              f"\n\teval_loss = {eval_loss},"
              f"\n\tcorrect_rate = {correct_rate}%")
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend(["train_losses", "test_losses"])
    plt.show()


def train_one_epoch(data_loader, net, device, criterion, optimizer, epoch_num):
    losses = []
    for index, (data, label) in enumerate(data_loader):
        numbers, fake_imgs = net(data.to(device))
        loss = compute_loss(numbers, label, data, fake_imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if index % cfg.LOG_PERIOD == 0:
            print(f"epoch [{epoch_num + 1}/{cfg.EPOCH}] of batch: {index}/{len(data_loader)}======loss: {loss.item()}")
    return np.mean(losses)


def eval_one_epoch(data_loader, net, device, criterion):
    losses = []
    correct_num, total = 0, 0
    with torch.no_grad():
        for index, (data, label) in enumerate(data_loader):
            numbers, fake_image = net(data.to(device))
            loss = compute_loss(numbers, label, data, fake_image)
            losses.append(loss.item())
            prop, judge = torch.max(numbers, dim=1)
            _, numers = torch.max(label.to(device), dim=1)
            correct_num += ((judge == numers).sum().item())
            total += label.size(0)
    return np.mean(losses), 100 * correct_num / total


def compute_loss(numbers, label, data, fake_images):
    criterion_number = torch.nn.CrossEntropyLoss()
    criterion_image = torch.nn.KLDivLoss()
    loss_number = criterion_number(numbers, label.to(numbers.device))
    loss_image = criterion_image(fake_images, data.to(fake_images.device))
    return loss_number * 0.7 + loss_image
