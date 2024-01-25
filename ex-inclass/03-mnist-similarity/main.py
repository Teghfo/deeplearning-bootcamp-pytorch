import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from tqdm import tqdm

batch_size = 32
learning_rate = 1e-3

train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=ToTensor())
test_ds = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=ToTensor())

train_dloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dloader = DataLoader(test_ds, batch_size=batch_size)


class NN(nn.Module):
    def __init__(self, in_feature: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_feature, 300)
        self.bn1 = nn.BatchNorm1d(300)
        self.fc2 = nn.Linear(300, 100)

    def forwad_once(self, x):
        x = self.flatten(x)
        output = self.bn1(self.fc1(x))
        output = F.relu()(output)
        output = self.fc2(output)
        return output

    def forward(self, x1, x2):
        out1 = self.forwad_once(x1)
        out2 = self.forwad_once(x2)
        return out1, out2


model = NN(28 * 28)


# Y/2 * (Dw) ^ 2 + (1-Y)/2 * (max(0, m-Dw) ^ 2)

class SimilarityLoss(nn.Module):
    pass


criterion = SimilarityLoss()
optimizer = optim.Adam(
    model.parameters(), lr=learning_rate)


# Train & validate Network
