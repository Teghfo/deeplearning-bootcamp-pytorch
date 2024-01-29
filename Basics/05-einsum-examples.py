import torch

# matrix mutiplication

A = torch.randn((5, 5))
B = torch.randn((5, 3))

vectorA = torch.arange(1, 10)
vectorB = torch.arange(11, 20)


print(torch.einsum("ik, kj -> ij", A, B))

# permute
print(torch.einsum("ij -> ji", A))

# diagonal
print(torch.einsum("ii -> i", A))

# trace
print(torch.einsum("ii ->", A))

# outer product
print(torch.einsum("i, j -> ij", vectorA, vectorB))

# inner product
print(torch.einsum("i, i -> ", vectorA, vectorB))

# element-wise product
print(torch.einsum("i, i -> i", vectorA, vectorB))
