import torch
import numpy as np

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# scaler(0-dim), vector(1-dim), matrix(2-dim), tensor(more than 2-dim)

print(torch.tensor(3, device=device))
print(torch.tensor([2, 3]))
print(torch.tensor([[2], [3]]))

# tensor n-dim ===> construct from tensors with n-1 dim
x = torch.tensor([[[3, 4, 5, 1], [4, 7, 56, -1]]],
                 dtype=torch.int32, device=device)
x.to(device)
print("size: ", x.size())
# print("shape: ", x.shape)
print(x.ndim)
print(x.dtype)

# methods for initialization tensors
print("\ncreate tensor using torch empty\n", torch.empty((2, 3)))
print("\ncreate tensor using torch zero\n", torch.zeros((2, 3)))
print("\ncreate tensor using torch ones\n", torch.ones((2, 3)))
print("\ncreate tensor using torch eye\n", torch.eye(4, 4))
# also see this methos: torch.ones_like, torch.zeros_like, torch.empty_like
print("\ncreate tensor using torch arange\n", torch.arange(2, 20, 2))
print("\ncreate tensor using torch linspace\n", torch.linspace(0.1, 2, 20))

# create tensor from numpy array
my_ndarr = np.random.randn(2, 2, 2)
my_tensor = torch.from_numpy(my_ndarr)
print(my_tensor)
my_ndarr[1, 1, 1] = 12
print(my_tensor)


x = torch.randn((2, 3))
print(type(x.numpy()))
