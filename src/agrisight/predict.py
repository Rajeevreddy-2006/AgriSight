"""Single-image prediction with calibrated confidence and abstention.

CLI:
  python -m agrisight.predict --ckpt runs/convnext_field_aug/best.pt leaf.jpg
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from PIL import Image

from .data import build_transforms
from .models import load_checkpoint


class Predictor:
    def __init__(self, ckpt_path: str, device: str | None = None, abstain_threshold: float = 0.6, tta: bool = True):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, meta = load_checkpoint(ckpt_path, self.device)
        self.classes = meta["classes"]
        self.info = meta.get("class_info", {})
        self.temperature = float(meta.get("temperature", 1.0))
        self.threshold = abstain_threshold
        self.tta = tta
        self.tf = build_transforms(meta["image_size"], meta["mean"], meta["std"], train=False)

    def display(self, label: str) -> str:
        i = self.info.get(label)
        return f"{i['crop']} - {i['disease']}" if i else label

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        x = self.tf(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)
        if self.tta:  # average with a horizontally flipped view
            logits = (logits + self.model(torch.flip(x, dims=[3]))) / 2
        probs = F.softmax(logits / self.temperature, dim=-1)[0].cpu()
        conf, idx = probs.topk(min(top_k, len(self.classes)))
        label = self.classes[idx[0]]
        info = self.info.get(label, {})
        return {
            "label": label,
            "display": self.display(label),
            "crop": info.get("crop"),
            "disease": info.get("disease"),
            "healthy": info.get("healthy"),
            "confidence": float(conf[0]),
            "abstain": float(conf[0]) < self.threshold,
            "top_k": [
                {"label": self.classes[i], "display": self.display(self.classes[i]), "prob": float(p)}
                for p, i in zip(conf.tolist(), idx.tolist())
            ],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()
    p = Predictor(args.ckpt, abstain_threshold=args.threshold)
    for path in args.images:
        print(path, json.dumps(p.predict(Image.open(path)), indent=2))


if __name__ == "__main__":
    main()
