import torch


OPTIMIZERS = {
    "adam": torch.optim.Adam
}


def build_optimizer(optimizer_params, cfg):
    return OPTIMIZERS[cfg.get("name")](optimizer_params, **cfg.get("params"))