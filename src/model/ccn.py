# Cascade convolutional network
import math
from typing import Dict, List, Tuple, Union, Iterable

import torch
import torch.nn as nn


ACTIVATIONS = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU
}


class ConvNormAct(nn.Module):
    def __init__(self, conv_params, use_bn, act) -> None:
        super().__init__()
        self.conv = nn.Conv2d(*conv_params)
        self.bn = nn.BatchNorm2d(conv_params[1]) if use_bn else nn.Identity()
        self.act = ACTIVATIONS[act[0]](*act[1]) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class ConvMP(nn.Module):
    def __init__(
        self,
        conv_params: Union[List, Tuple],
        pool_param: Union[List, Tuple]
    ) -> None:
        super().__init__()
        self.conv = ConvNormAct(conv_params[:-2], *conv_params[-2:])
        self.pool = nn.MaxPool2d(*pool_param) if len(pool_param) > 0 \
                    else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(x))


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

class CCN(nn.Module):
    def __init__(
        self,
        backbone_cfg: Iterable[Union[List, Tuple]],
        head_cfg: Union[List, Tuple],
        input_size: Union[List[int], Tuple[int, int]], # w, h
        in_channels: int = 3
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        layers = []
        for conv_params, pool_param in backbone_cfg:
            conv_params = [in_channels] + list(conv_params)
            layers.append(ConvMP(conv_params, pool_param))
            in_channels = conv_params[1]
        self.backbone = nn.Sequential(*layers)
        if not head_cfg.get("in_channels"):
            _, c, h, w = self.backbone(torch.rand(1, self.in_channels, input_size[1], input_size[0])).shape
            head_cfg["in_channels"] = c * h * w
        self.head = Head(**head_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x

def build_ONet(input_size, in_channels, out_channels, weights=None):
    backbone_cfg = (
        ((32, (3, 3), 1, 0, True, ("relu", ())), ((3, 3), 2, 0, 1, False, True)),
        ((64,(3, 3), 1, 0, True, ("relu", ())), ((3, 3), 2, 0, 1, False, True)),
        ((64, (3, 3), 1, 0, True, ("relu", ())), ((3, 3), 2, 0, 1, False, True)),
        ((128, (2, 2), 1, 0, True, ("relu", ())), ())
    )
    head_cfg = {
        "out_channels": out_channels
    }
    net = CCN(backbone_cfg, head_cfg, input_size, in_channels)
    if weights:
        net.load_state_dict(torch.load(weights))
        print("loaded")
    return net