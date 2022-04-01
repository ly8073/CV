import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


from MINIST.models.model_zoo import NetWork
import MINIST.configs as cfg


def train(train_set, test_set):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    nets = NetWork(1, 16).to(device)
    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE)

    optimizer = torch.optim.Adam(nets.parameters(), lr=cfg.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    train_losses = []
    for i in range(cfg.EPOCH):
        loss = train_one_epoch(train_loader, nets, device, criterion, optimizer, i)
        train_losses.append(loss)
        print(f"epoch [{i} / {cfg.EPOCH}] trained done. loss={loss}")
        if i % cfg.CHECKPOINT == (cfg.CHECKPOINT - 1):
            torch.save(nets, os.path.join(cfg.FOLDER, f"checkpoints_{i}.pkl"))
    plt.plot(train_losses)
    plt.show()


def train_one_epoch(train_loader, net, device, criterion, optimizer, epoch_num):
    losses = []
    for index, (train_data, train_label) in enumerate(train_loader):
        y = net(train_data.to(device))
        loss = criterion(y, train_label.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if index % cfg.LOG_PERIOD == 0:
            print(f"epoch [{epoch_num + 1}/{cfg.EPOCH}] of batch: {index}/{len(train_loader)}======loss: {loss.item()}")
    return np.sum(losses)



