import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.s2 = nn.AvgPool2d(kernel_size=(2, 2))
        self.c3 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.s4 = nn.AvgPool2d(kernel_size=(2, 2))
        self.c5 = nn.Conv2d(16, 120, kernel_size=(5, 5))

        self.fc = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        x = F.tanh(self.c1(x))
        x = self.s2(x)
        x = F.tanh(self.c3(x))
        x = self.s4(x)
        x = F.tanh(self.c5(x))
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc(x))
        x = self.out(x)
        return x


vgg16 = models.vgg16()
