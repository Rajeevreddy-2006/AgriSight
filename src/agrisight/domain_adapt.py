"""Unsupervised domain adaptation from lab photos to field photos.

* AdaBN: re-estimate BatchNorm statistics on unlabelled field images (no training).
  Used through `agrisight.evaluate --adabn-manifest field_train`.
* DANN: adversarial training with a gradient reversal layer so the backbone learns
  features a domain classifier cannot tell apart (lab vs field). Field labels are
  never used.

Example:
  python -m agrisight.domain_adapt --init runs/convnext_field/best.pt --run-name convnext_dann
"""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def adabn(model: nn.Module, loader, device) -> nn.Module:
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        warnings.warn("Model has no BatchNorm layers (ConvNeXt/ViT use LayerNorm), so AdaBN has no effect. "
                      "Try resnet50 or efficientnet_b0.")
        return model
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over all target batches
    model.train()
    for x, _ in tqdm(loader, desc="AdaBN", leave=False):
        model(x.to(device))
    model.eval()
    return model


def endless(loader):
    """Re-iterate a DataLoader forever (fresh augmentation each pass, no batch caching)."""
    while True:
        yield from loader


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


class DomainHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 1))

    def forward(self, feats, lambd):
        return self.net(GradReverse.apply(feats, lambd)).squeeze(-1)


def main():
    from .config import get_device, load_config, manifest_path, set_seed
    from .data import ManifestDataset, build_transforms
    from .engine import make_loader, predict_logits
    from .models import feature_dim, features_and_logits, load_checkpoint, save_checkpoint

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="source-trained checkpoint to start from")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--domain-weight", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, help="debug: stop each epoch after N steps")
    ap.add_argument("--device")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(args.device)
    model, meta = load_checkpoint(args.init, device)
    classes = meta["classes"]
    size, mean, std = meta["image_size"], meta["mean"], meta["std"]
    nw = cfg["data"]["num_workers"]

    src = ManifestDataset(manifest_path(cfg, "lab_train"), classes, build_transforms(size, mean, std, True, meta.get("augment", "basic")))
    tgt = ManifestDataset(manifest_path(cfg, "field_train"), classes, build_transforms(size, mean, std, True, "basic"))
    val = ManifestDataset(manifest_path(cfg, "lab_val"), classes, build_transforms(size, mean, std, False))
    src_loader = make_loader(src, args.batch_size, True, nw, device)
    tgt_loader = make_loader(tgt, args.batch_size, True, nw, device)
    val_loader = make_loader(val, args.batch_size * 2, False, nw, device)

    head = DomainHead(feature_dim(model)).to(device)
    optimizer = torch.optim.AdamW(
        [{"params": model.parameters(), "lr": args.lr}, {"params": head.parameters(), "lr": args.lr * 10}],
        weight_decay=0.01,
    )
    steps_per_epoch = min(len(src_loader), args.max_steps or len(src_loader))
    total = args.epochs * steps_per_epoch
    step = 0
    run_dir = Path(cfg["paths"]["runs_dir"]) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        tgt_iter = endless(tgt_loader)
        cls_sum = dom_sum = dom_acc = n = 0.0
        for i, (xs, ys) in enumerate(tqdm(src_loader, desc=f"DANN epoch {epoch}", leave=False, total=steps_per_epoch)):
            if i >= steps_per_epoch:
                break
            xt, _ = next(tgt_iter)  # field labels are ignored
            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
            p = step / max(1, total)
            lambd = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0  # ramps 0 -> 1

            fs, logits = features_and_logits(model, xs)
            ft, _ = features_and_logits(model, xt)
            cls_loss = F.cross_entropy(logits, ys)
            dom_logits = head(torch.cat([fs, ft]), lambd)
            dom_labels = torch.cat([torch.zeros(len(fs)), torch.ones(len(ft))]).to(device)
            dom_loss = F.binary_cross_entropy_with_logits(dom_logits, dom_labels)
            loss = cls_loss + args.domain_weight * dom_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            cls_sum += cls_loss.item()
            dom_sum += dom_loss.item()
            dom_acc += ((dom_logits > 0).float() == dom_labels).float().mean().item()
            n += 1

        val_logits, val_labels = predict_logits(model, val_loader, device, desc="val")
        val_acc = (val_logits.argmax(1) == val_labels).float().mean().item()
        # Domain accuracy near 0.5 means lab and field features have become hard to tell apart.
        print(f"epoch {epoch} cls_loss {cls_sum / n:.4f} dom_loss {dom_sum / n:.4f} "
              f"dom_acc {dom_acc / n:.3f} lab_val_acc {val_acc:.4f}")

    save_checkpoint(run_dir / "best.pt", model, {**meta, "adaptation": "dann", "temperature": 1.0})
    print(f"Saved {run_dir / 'best.pt'}. Evaluate with:\n  python -m agrisight.evaluate --ckpt {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
