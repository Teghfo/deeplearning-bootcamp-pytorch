import torch
import torch.nn as nn


class SeperableConv2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, *args,  groups=in_channels, **kwargs)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


model = SeperableConv2d(3, 64, kernel_size=(3, 3), padding=(1, 1))
x = torch.randn(size=(1, 3, 224, 224))
print(model(x).size())
