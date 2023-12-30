import torch

x = torch.tensor(range(10))
y = torch.tensor(range(2, 48, 5))

# y = x * w
# z = torch.mul(x, w)

# y = x + w
# z = torch.add(x, w)
# print(y)
# print(z)
print("x", x)
print("y", y)
# print(x.add(2))

# x.add_(2) ## inplace (every methode with underlying underscore) (x += )
# print("x + 2", x)

# universal functions
print("x ^ 3: ", x.pow(3))
print("sin(x): ", x.sin())
print("tan(x): ", x.tan())


# tensor aggregate operations
# x = torch.arange(10)
# x = torch.tensor([[1, 2], [3, 4]])
x = torch.tensor([[[1, 2], [3, 4]], [[6, 8], [-1, 5]]])
print("shape x", x.shape)
sum_x = torch.sum(
    x, dim=0
)
print("sum_x 0: ", sum_x)
# print("sum 1", torch.sum(

#     x, dim=1
# ))
# print("sum 2", torch.sum(
#     x, dim=2
# ))
torch.min(x, dim=0)
torch.argmin(x, dim=0)
torch.max(x, dim=0)
torch.argmax(x, dim=0)
torch.mean(x.float(), dim=0)

# casting dtype
print(x.long().dtype)
print(x.half().dtype)
print(x.bool())

# comparision
x = torch.arange(10)
print(x > 2)
print(x[x > 2])


torch.random.manual_seed(42)

mat1 = torch.rand((3, 6))
mat2 = torch.rand((6, 2))
print("mat1 @ mat2", mat1 @ mat2)
print("mat1 @ mat2", torch.mm(mat1, mat2))


# slicing and indexing
x = torch.rand((32, 784))


# reshape: view, reshape, squeeze, unsqueeze
x = torch.rand((2, 4))
w = x.t()
print(w.is_contiguous())
# print(w.view(1, 8))
print(w.reshape(1, 8))
# print(x.view(2, 5))
# print(x.reshape(2, 50))

x = torch.arange(10)
y = x.reshape(1, 10)
print(y)
print(y.squeeze(0).shape)
print(y.squeeze())
print(y.unsqueeze(0).shape)
print(y.unsqueeze(2))
