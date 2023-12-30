import torch

# y = 5 * x + 2
x = torch.linspace(-1, 1, 1000)
y = 5 * x + 2
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)


def model(input, w, b):
    return w * input + b


def criterion(y_pred, y_true):
    return torch.pow(y_pred - y_true, 2).mean()


epoch = 100
batch_size = 32
learning_rate = 0.01


for i in range(epoch):
    for batch in range((len(x) // batch_size)):
        input_batch = x[batch * batch_size: (batch+1) * batch_size]
        y_true = y[batch * batch_size: (batch+1) * batch_size]
        y_pred = model(input_batch, w, b)
        loss = criterion(y_pred, y_true)
        if w.grad:
            w.grad.zero_()
            b.grad.zero_()

        loss.backward()

        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
        if i % 10 == 0:
            print(
                f"Epoch {i}/{epoch}, step {batch}/{len(x) // batch_size} loss: {loss.item()}, w: {w.item()}, b: {b.item()}")
