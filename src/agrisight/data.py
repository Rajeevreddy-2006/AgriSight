"""Datasets and image transforms, including field-style augmentation."""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def read_manifest(path: str | Path) -> list[tuple[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["path"], r["label"]) for r in csv.DictReader(f)]


class ManifestDataset(Dataset):
    """Images listed in one or more manifest CSVs (columns: path,label)."""

    def __init__(self, manifests, classes: list[str], transform=None, limit: int | None = None):
        if isinstance(manifests, (str, Path)):
            manifests = [manifests]
        rows = [row for m in manifests for row in read_manifest(m)]
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = [(p, self.class_to_idx[l]) for p, l in rows if l in self.class_to_idx]
        if limit:
            self.samples = self.samples[:limit]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        return (self.transform(img) if self.transform else img), y


def build_transforms(image_size: int, mean, std, train: bool, augment: str = "basic"):
    normalize = [T.ToTensor(), T.Normalize(mean, std)]
    if not train or augment == "none":
        resize = int(round(image_size / 0.875))
        return T.Compose([T.Resize(resize), T.CenterCrop(image_size), *normalize])
    if augment == "basic":
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            *normalize,
        ])
    if augment == "field":
        # Simulates phone photos in a field: off-centre framing, partial leaves,
        # changing light, blur, rotation and occlusion.
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.25, 1.0), ratio=(0.6, 1.6)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.TrivialAugmentWide(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            T.RandomGrayscale(p=0.05),
            *normalize,
            T.RandomErasing(p=0.25),
        ])
    raise ValueError(f"Unknown augment mode: {augment}")
