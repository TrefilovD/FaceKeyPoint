import torch

SCHEDULERS = {
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
    "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
    "CosineAnnealingLR":  torch.optim.lr_scheduler.CosineAnnealingLR
}

def build_scheduler(optimizer, cfg):
    if cfg:
        return SCHEDULERS[cfg.get("name")](optimizer, **cfg.get("params"))
    return None