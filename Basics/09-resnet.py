import torch
import torch.nn as nn
import torch.nn.functional as F


class ResUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bot_layer: bool = False):
        super().__init__()
        self.bot_layer = bot_layer
        self.cn1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      stride=2 if bot_layer else 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.cn2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        self.cn_bl = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.cn1(x)
        out = self.cn2(out)
        if self.bot_layer:
            x = self.cn_bl(x)
        out += x
        out = F.relu(out)
        return out


class ResNet34(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, 64,
                      stride=2, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.res_block = nn.ModuleList()
        blocks = [[64] * 3, [128] * 4, [256] * 6, [512] * 3]
        for indx, block in enumerate(blocks):
            temp_module = nn.ModuleList()
            if not indx:
                for i in block:
                    temp_module.append(ResUnit(i, i))
            else:
                temp_module.append(
                    ResUnit(block[0]//2, block[0], bot_layer=True))
                for i in block[1:]:
                    temp_module.append(ResUnit(i, i))
            self.res_block.append(temp_module)

        self.out_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 1)),
            nn.Flatten(),
            nn.Linear(1024, 1000)
        )

    def forward(self, x):
        x = self.input_block(x)
        for sub_blocks in self.res_block:
            for layer in sub_blocks:
                x = layer(x)

        x = self.out_block(x)
        return x


# model = ResUnit(64, 128, bot_layer=True)
# x = torch.randn(size=(1, 64, 32, 32))
# print(model(x).size())
# print(sum([p.numel() for p in model.parameters()]))
model = ResNet34(3)
x = torch.randn(size=(2, 3, 32, 32))
print(model(x).size())
