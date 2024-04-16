from torch.utils.data import DataLoader


def build_dataloader(dataset, cfg):
    return DataLoader(
        dataset = dataset,
        batch_size = cfg.get("batch_size"),
        shuffle = cfg.get("shuffle"),
        sampler = None, # build sampler
        num_workers = cfg.get("num_workers"),
        collate_fn = None, # build collate
    )