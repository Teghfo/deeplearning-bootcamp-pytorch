import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels=32, out_channels=64,
                              kernel_size=(3, 3), stride=1, padding=(1, 1))

        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=128,
                              kernel_size=(3, 3), stride=1, padding=(1, 1))


# model = CNN()
# for p in model.parameters():
#     print("parameters: ", p.size())
#     print(p.numel())

model = nn.Conv2d(in_channels=32, out_channels=64,
                  kernel_size=(3, 3), stride=1, padding=(1, 1))


x = torch.randn(size=(1, 32, 5, 5))
# print(list(model.parameters())[0][0])

# print(sum(p.numel() for p in model.parameters()))


class MinorGoogleNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.cn1 = BasicConv2d(in_channels, 64,
                               kernel_size=(7, 7), stride=2, padding=(3, 3))
        self.s1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=(1, 1))

        self.inception1 = Inception(64, 64, 96, 128, 12, 32, 32)
        self.inception2 = Inception(256, 384, 192, 384, 48, 128, 128)

        self.gavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cn1(x)
        x = self.s1(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.gavgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels: int, b1_11_ch: int, b2_11_ch: int, b2_33_ch: int, b3_11_ch: int, b3_55_ch: int, b4_11_ch: int):
        super().__init__()
        self.b1 = BasicConv2d(in_channels, b1_11_ch, kernel_size=(
            1, 1), stride=1, padding=(0, 0))

        self.b2 = nn.Sequential(BasicConv2d(in_channels, b2_11_ch, kernel_size=(1, 1), stride=1, padding=(0, 0)),
                                BasicConv2d(b2_11_ch, b2_33_ch, kernel_size=(
                                    3, 3), stride=1, padding=(1, 1))
                                )

        self.b3 = nn.Sequential(BasicConv2d(in_channels, b3_11_ch, kernel_size=(1, 1), stride=1, padding=(0, 0)),
                                BasicConv2d(b3_11_ch, b3_55_ch, kernel_size=(
                                    5, 5), stride=1, padding=(2, 2))
                                )
        self.b4 = nn.Sequential(nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=(1, 1)),
                                BasicConv2d(in_channels, b4_11_ch, kernel_size=(
                                    1, 1), stride=1, padding=(0, 0))
                                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)

        x = torch.cat([b1, b2, b3, b4], 1)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()

        self.cn = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cn(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


# model = Inception(3, 64, 96, 128, 12, 32, 32)
# x = torch.randn(size=(1, 3, 5, 5))
# print(model(x).size())


# model = nn.AdaptiveAvgPool2d((2, 2))
# x = torch.randn(size=(1, 32, 5, 5))
# # for i in range(32):
# # print(x[0, i].mean())
# #
# model2 = nn.AvgPool2d((4, 4), stride=1)
# print(model(x))
# print(model2(x))


# print(model(x).size())


model = MinorGoogleNet(3, 100)
x = torch.randn(size=(1, 3, 32, 32))
print(model(x))

