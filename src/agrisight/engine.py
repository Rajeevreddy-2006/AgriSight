"""Shared helpers for running a model over a DataLoader."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def predict_logits(model, loader, device, desc: str = "eval"):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in tqdm(loader, desc=desc, leave=False):
        all_logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)
