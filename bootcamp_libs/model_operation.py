from typing import List
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer


def per_epoch(data_loader: DataLoader, model: Module, optimizer: Optimizer, criterion, device) -> List[float]:
    losses = []

    for i, (data, labels) in tqdm(enumerate(data_loader)):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        predictions = model(data)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses
