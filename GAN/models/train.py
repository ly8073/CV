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
from torch.nn.functional import one_hot

from GAN.models.model_zoo import GanForMinist, Discriminator
import MINIST.configs as cfg


class Trainer:
    def __init__(self, epoch=None):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if epoch is not None:
            try:
                check_points = os.path.join(cfg.CHECKPOINT_FOLDER, f"checkpoints_{epoch}.pkl")
                dis_criminator_checkpoints = os.path.join(cfg.CHECKPOINT_FOLDER, f"discriminator_{epoch}.pkl")
                self.nets = torch.load(check_points).to(self.device)
                self.discriminator = torch.load(dis_criminator_checkpoints).to(self.device)
            except Exception:
                self.nets = GanForMinist(1, 16, 32).to(self.device)
                self.discriminator = Discriminator(784, 1).to(self.device)
        else:
            self.nets = GanForMinist(1, 16, 32).to(self.device)
            self.discriminator = Discriminator(784, 1).to(self.device)
        self.correct_rate = 0.80
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                        lr=cfg.DISCRIMINATOR_LEARNING_RATE)
        self.optimizer_decode = torch.optim.Adam(self.nets.decode.parameters(), lr=cfg.LEARNING_RATE)
        self.optimizer_encode = torch.optim.Adam(self.nets.encode.parameters(), lr=cfg.LEARNING_RATE)

    def train(self, train_set, test_set):
        train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE)
        test_loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE)
        train_losses, test_losses, correct_rates = [], [], []
        for i in range(cfg.EPOCH):
            discriminator_loss = self.train_discriminator(train_loader)
            loss = self.train_one_epoch(train_loader, i)
            train_losses.append(loss)
            eval_loss, correct_rate = self.eval_one_epoch(test_loader)
            correct_rates.append(correct_rate)
            test_losses.append(eval_loss)
            if i % cfg.CHECKPOINT == (cfg.CHECKPOINT - 1):
                if not os.path.exists(cfg.CHECKPOINT_FOLDER):
                    os.mkdir(cfg.CHECKPOINT_FOLDER)
                torch.save(self.nets, os.path.join(cfg.CHECKPOINT_FOLDER, f"checkpoints_{i}.pkl"))
                torch.save(self.discriminator, os.path.join(cfg.CHECKPOINT_FOLDER, f"discriminator_{i}.pkl"))
            print(f"epoch [{i + 1} / {cfg.EPOCH}] trained done. "
                  f"\n\tdiscirminator_loss = {discriminator_loss}"
                  f"\n\tloss={loss}, "
                  f"\n\teval_loss = {eval_loss},"
                  f"\n\tcorrect_rate = {correct_rate}%")
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.plot(test_losses)
        plt.legend(["train_losses", "test_losses"])
        plt.subplot(1, 2, 2)
        plt.plot(correct_rates)
        plt.show()

    def train_one_epoch(self, data_loader, epoch_num):
        losses = []
        for index, (data, label) in enumerate(data_loader):
            data = data.to(self.device)
            label = label.to(self.device)
            numbers, fake_imgs = self.nets(data, label)
            fake_dis = self.discriminator(fake_imgs)
            loss_number, loss_image = compute_loss(numbers, label, data, fake_imgs)
            loss = loss_image - torch.sum(fake_dis)
            if self.correct_rate <= 0.98:
                self.optimizer_encode.zero_grad()
                loss_number.backward()
                self.optimizer_encode.step()
            self.optimizer_decode.zero_grad()
            loss.backward()
            self.optimizer_decode.step()
            losses.append(loss.item())
            if index % cfg.LOG_PERIOD == 0:
                print(
                    f"epoch [{epoch_num + 1}/{cfg.EPOCH}] of batch: {index}/{len(data_loader)}======loss: {loss.item()},"
                    f" loss_number: {loss_number.item()}")
        return np.mean(losses)

    def train_discriminator(self, data_loader):
        losses = []
        for index, (data, label) in enumerate(data_loader):
            data = data.to(self.device)
            real_dis = self.discriminator(data)
            random_input = generate_random_number(data.shape[0]).to(self.device)
            fake_image = self.nets.decode(random_input)
            fake_dis = self.discriminator(fake_image)
            loss = torch.sum(fake_dis - real_dis)
            losses.append(loss.item())
            self.discriminator_optimizer.zero_grad()
            loss.backward()
            self.discriminator_optimizer.step()
        return np.mean(losses)

    def eval_one_epoch(self, data_loader):
        losses = []
        correct_num, total = 0, 0
        with torch.no_grad():
            for index, (data, label) in enumerate(data_loader):
                data = data.to(self.device)
                label = label.to(self.device)
                numbers, fake_image = self.nets(data)
                fake_dis = self.discriminator(fake_image)
                loss_number, loss_image = compute_loss(numbers, label, data, fake_image)
                loss = loss_image - torch.sum(fake_dis)
                losses.append(loss.item())
                prop, judge = torch.max(numbers, dim=1)
                _, numers = torch.max(label, dim=1)
                correct_num += ((judge == numers).sum().item())
                total += label.size(0)
        self.correct_rate = correct_num / total
        return np.mean(losses), 100 * self.correct_rate


def compute_loss(numbers, label, data, fake_images):
    criterion_number = torch.nn.CrossEntropyLoss()
    criterion_image = torch.nn.KLDivLoss()
    loss_number = criterion_number(numbers, label)
    loss_image = criterion_image(fake_images, data) + criterion_image(data, fake_images)
    return loss_number, loss_image


def generate_random_number(batch_num):
    random_number = torch.randint(0, 10, (batch_num, ))
    return one_hot(random_number, num_classes=10).float()
