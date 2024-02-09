from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/tensorboard_tutorial_{timestamp}')


class ConvNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(10, 5, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(24 * 24 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.layers(x)


dataset = MNIST(root="../notebooks/data", download=True,
                transform=transforms.ToTensor())
trainloader = DataLoader(
    dataset, batch_size=32, shuffle=True, num_workers=1)

convnet = ConvNeuralNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(convnet.parameters(), lr=1e-4)

epochs_num = 5

for epoch in range(epochs_num):

    print(f'Epoch: {epoch+1}/{epochs_num}')
    current_loss = 0.0
    loss_idx = 0

    for i, (inputs, targets) in enumerate(trainloader):

        if epoch == 0 and i == 0:
            writer.add_graph(convnet, input_to_model=inputs, verbose=False)

        if i == 0:
            writer.add_image("Example input", inputs[0], global_step=epoch)

        optimizer.zero_grad()

        outputs = convnet(inputs)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        current_loss += loss.item()
        writer.add_scalar("Loss/perMiniBatch", current_loss, loss_idx)
        loss_idx += 1
        if i % 500 == 499:
            print(f'Loss after mini-batch {i+1}: {(current_loss / 500):.3f}')
            current_loss = 0.0

    writer.add_scalar("Loss/perEpochs", current_loss, epoch)
