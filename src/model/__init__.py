from .ccn import build_ONet
from .mobilenet_v2 import build_MobileNetV2
from .resnet import build_resnet18

from utils.model_utils import freeze_modules


NETWORKS = {
    "onet": build_ONet,
    "mobilenet_v2": build_MobileNetV2,
    "resnet18": build_resnet18
}

def build_network(cfg):
    net = NETWORKS[cfg.get("name")](**cfg.get("params"))
    if cfg.get("freeze"):
        modules_2_freeze = cfg.get("freeze")
        freeze_modules(net, modules_2_freeze)
    return net