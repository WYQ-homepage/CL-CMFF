"""Explicit random streams for data loading and augmentation."""
import hashlib
import random

import numpy as np
import torch
from torch.utils.data import get_worker_info


def data_seed(seed, rank=0, stage=0, epoch=0, split="train"):
    """Derive a stable Torch seed without depending on Python's hash seed."""
    key = f"{int(seed)}:{int(rank)}:{int(stage)}:{int(epoch)}:{split}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little") % (2 ** 63)


def seed_augmentations(dataset, seed, seed_globals=False):
    """Give each Albumentations pipeline its own reproducible random stream."""
    if seed_globals:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32))
    for offset, name in enumerate(("unified_transform", "rgb_transform", "ir_transform", "sar_transform")):
        transform = getattr(dataset, name, None)
        if transform is not None:
            transform.set_random_seed((int(seed) + 1009 * offset) % (2 ** 32))


def seed_worker(worker_id):
    """Seed global RNGs and the worker's independent augmentation instances."""
    worker = get_worker_info()
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if worker is not None:
        seed_augmentations(worker.dataset, seed)


def make_data_generator(seed, rank=0, split="train"):
    return torch.Generator().manual_seed(data_seed(seed, rank=rank, split=split))


def set_loader_epoch(loader, sampler, seed, rank, stage, epoch):
    """Separate data randomness by rank, stage and epoch, including zero workers."""
    if loader.persistent_workers:
        raise ValueError("Stage/epoch reseeding requires persistent_workers=False")
    sampler.set_epoch(epoch)
    if loader.generator is None:
        raise ValueError("Training DataLoader must have an independent generator")
    current_seed = data_seed(seed, rank=rank, stage=stage, epoch=epoch)
    loader.generator.manual_seed(current_seed)
    # Zero-worker CustomAffine/DSM augmentation runs in the main process.
    seed_augmentations(loader.dataset, current_seed, seed_globals=loader.num_workers == 0)
