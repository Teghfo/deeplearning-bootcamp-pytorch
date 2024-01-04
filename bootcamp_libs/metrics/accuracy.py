import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_accuracy(data_loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    # # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for data, label in data_loader:

            data = data.to(device=device)
            label = label.to(device=device)

            # Forward pass
            output = model(data)
            _, predictions = output.max(1)

            num_correct += (predictions == label).sum()

            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples
