import torch

import torch.nn as nn
from torchvision.models import MobileNetV2 as mobilenet_v2



class Head(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int = 10
        ) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_channels, 256)
        self.drop5 = nn.Dropout(0.25)
        self.act = nn.PReLU(256)
        self.linear2 = nn.Linear(256, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.act(self.drop5(self.linear1(x)))
        x = self.linear2(x)
        return x


class MobileNetV2(torch.nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model = mobilenet_v2()
        self.model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier = Head(1280, out_channels)

    def forward(self, x):
        return self.model(x)


def build_MobileNetV2(in_channels, out_channels, weights=None):

    net = MobileNetV2(in_channels, out_channels)
    if weights:
            net.load_state_dict(torch.load(weights))
    return net