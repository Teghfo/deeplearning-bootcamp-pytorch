import torch


def find_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cp"
