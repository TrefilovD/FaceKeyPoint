import math

import torch


class L2Loss(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, pred, gt, mask):
        gt = gt * mask
        pred = pred * mask
        x = torch.abs(gt - pred)
        return torch.norm(x, p=2) ** 2 / (x.numel() + 1e-7)


class WingLoss(torch.nn.Module):
    def __init__(self, w, eps=1e-8) -> None:
        super().__init__()
        assert w > 0
        self.w = w
        self.eps = eps
        self.C = w - w * math.log((1 + w) / eps)

    def forward(self, pred, gt, mask):
        gt = gt * mask
        pred = pred * mask
        x = torch.abs(gt - pred)
        loss1 = self.w * torch.log(1 + x[x < self.w] / self.eps)
        C = self.w - self.w * math.log(1 + self.w / self.eps)
        loss2 = x[x >= self.w] - C
        return (loss1.sum() + loss2.sum()) / (x.numel() + 1e-7)


CRITERIONS = {
    "L2Loss": L2Loss,
    "WingLoss": WingLoss,
    "L1Loss": torch.nn.L1Loss,
    "SmoothL1Loss": torch.nn.SmoothL1Loss
}

def build_criterion(cfg):
    return CRITERIONS[cfg.get("name")](**cfg.get("params"))