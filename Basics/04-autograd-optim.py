import torch
from torch import nn, optim


# y = 5 * x + 2
x = torch.linspace(-1, 1, 1000)
y = 5 * x + 2
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)


def model(input, w, b):
    return w * input + b


epoch = 100
batch_size = 32
learning_rate = 0.01

criterion = nn.MSELoss()
optimizer = optim.SGD([w, b], lr=learning_rate)


for i in range(epoch):
    for batch in range((len(x) // batch_size)):
        input_batch = x[batch * batch_size: (batch+1) * batch_size]
        y_true = y[batch * batch_size: (batch+1) * batch_size]
        y_pred = model(input_batch, w, b)
        loss = criterion(y_pred, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(
                f"Epoch {i}/{epoch}, step {batch}/{len(x) // batch_size} loss: {loss.item()}, w: {w.item()}, b: {b.item()}")
