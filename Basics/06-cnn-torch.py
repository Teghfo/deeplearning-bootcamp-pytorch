import torch
import torch.nn as nn


m = nn.Conv2d(in_channels=1, out_channels=64,
              kernel_size=3, stride=2)
# trainable parameters ===> (3 * 3 * 1) * 64 + 64
print("parameters:", sum(p.numel() for p in m.parameters()))


data = torch.randn((32, 1, 28, 28))
out = m(data)
print(out.size())
