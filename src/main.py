import argparse
import os
import os.path as osp
import yaml

from pprint import pprint
from yaml import load, SafeLoader, FullLoader
from clearml import Task, Logger

from data import build_dataset
from model import build_network
from modules.actions import train, test
from modules.optimizer import build_optimizer
from modules.loss import build_criterion
from modules.scheduler import build_scheduler
from modules.dataloader import build_dataloader


def setup(mode, cfg, task, logger):
    os.makedirs(cfg.get("task_info").get("save_dir"), exist_ok=True)
    save_dir = osp.join(cfg.get("task_info").get("save_dir"), cfg.get("task_info").get("name"))
    os.makedirs(save_dir, exist_ok=True)

    model = build_network(cfg.get("network"))
    criterion = build_criterion(cfg.get("criterion"))


    if mode == "train":
        learning_params = filter(lambda p: p.requires_grad, model.parameters())
        print(f"{len(list(learning_params))}/{len(list(model.parameters()))} will be optimize")
        optimizer = build_optimizer(filter(lambda p: p.requires_grad, model.parameters()), cfg.get("optimizer"))
        scheduler = build_scheduler(optimizer, cfg.get("scheduler"))
        train_cfg = cfg.get("train_cfg")
        train_dataset = build_dataset("train", cfg.get("train_dataset"))
        train_dataloader = build_dataloader(train_dataset, cfg.get("train_dataloader"))
        val_dataset = build_dataset("val", cfg.get("val_dataset"))
        val_dataloader = build_dataloader(val_dataset, cfg.get("val_dataloader"))
        train(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            criterion,
            scheduler,
            train_cfg,
            save_dir,
            logger,
            task
        )
    elif mode == "test":
        val_dataset = build_dataset("test", cfg.get("test_dataset"))
        val_dataloader = build_dataloader(val_dataset, cfg.get("test_dataloader"))
        test(
            model,
            val_dataloader,
            criterion,
            cfg.get("test_cfg").get("device"),
            save_dir,
            task,
            logger
        )
    else:
        raise f"Unknown launch mode \"{mode}\""




def parse():
    parser = argparse.ArgumentParser("Parser config files")
    parser.add_argument("-m", "--mode", type=str, action="store", choices=["train", "test"], required=True, help="mode of launch")
    parser.add_argument("-c", "--config", type=str, action="store", required=True, help="path to config yaml")
    args = parser.parse_args()
    return args

def parse_yaml(path):
    with open(path, "r") as fr:
        return yaml.load(fr, Loader=FullLoader)
    

def get_tags(cfg):
    tags_name = ["network", "optimizer", "criterion", "scheduler"]
    tags = []
    for tag_name in tags_name:
        if cfg.get(tag_name) and cfg.get(tag_name).get("name"):
            tags.append(cfg[tag_name].get("name"))
    return tags


if __name__ == "__main__":
    args = parse()
    cfg = parse_yaml(args.config)
    task = Task.init(
        project_name='FaceKeypointDetection',
        task_name=cfg.get("task_info").get("name"),
        tags=[args.mode] + get_tags(cfg)
    )
    logger = Logger.current_logger()
    task.upload_artifact(name='config.yaml', artifact_object=args.config)
    task.upload_artifact(name='mode', artifact_object=args.mode)
    setup(args.mode, cfg, task, logger)
    task.close()