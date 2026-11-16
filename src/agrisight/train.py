"""Train an image classifier on lab (and optionally field) images.

Examples:
  # Baseline: train on PlantVillage only
  python -m agrisight.train --run-name convnext_field

  # Compare backbones / augmentation
  python -m agrisight.train --run-name resnet50_basic --model resnet50 --augment basic

  # Upper bound: also train on labelled field images
  python -m agrisight.train --run-name oracle --train-manifests lab_train,field_train

  # Knowledge distillation into a small phone-friendly student
  python -m agrisight.train --run-name student_mnv3 --model mobilenetv3_large_100 \
      --teacher runs/convnext_field/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from .config import get_device, load_config, manifest_path, set_seed
from .data import ManifestDataset, build_transforms
from .engine import make_loader, predict_logits
from .models import build_model, load_checkpoint, model_norm, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--run-name", required=True)
    p.add_argument("--model")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--augment", choices=["none", "basic", "field"])
    p.add_argument("--train-manifests", default="lab_train", help="comma-separated manifest names")
    p.add_argument("--val-manifest", default="lab_val")
    p.add_argument("--teacher", help="checkpoint of a teacher model for knowledge distillation")
    p.add_argument("--kd-alpha", type=float, default=0.7, help="weight of the distillation loss")
    p.add_argument("--kd-temp", type=float, default=4.0)
    p.add_argument("--limit", type=int, help="debug: use at most N training images")
    p.add_argument("--device")
    return p.parse_args()


def kd_loss(student_logits, teacher_logits, T: float):
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    tc = cfg["train"]
    for key in ("model", "epochs", "batch_size", "lr", "augment"):
        if getattr(args, key) is not None:
            tc[key] = getattr(args, key)

    set_seed(cfg["seed"])
    device = get_device(args.device)
    run_dir = Path(cfg["paths"]["runs_dir"]) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_classes = json.loads((Path(cfg["paths"]["manifest_dir"]) / "classes.json").read_text())
    classes, class_info = meta_classes["classes"], meta_classes["info"]

    model = build_model(tc["model"], len(classes), pretrained=tc["pretrained"], drop_rate=tc["drop_rate"]).to(device)
    mean, std = model_norm(model)
    size = cfg["data"]["image_size"]

    train_files = [manifest_path(cfg, m) for m in args.train_manifests.split(",")]
    train_ds = ManifestDataset(train_files, classes, build_transforms(size, mean, std, True, tc["augment"]), limit=args.limit)
    val_ds = ManifestDataset(manifest_path(cfg, args.val_manifest), classes, build_transforms(size, mean, std, False))
    nw = cfg["data"]["num_workers"]
    train_loader = make_loader(train_ds, tc["batch_size"], True, nw, device)
    val_loader = make_loader(val_ds, tc["batch_size"] * 2, False, nw, device)
    print(f"device={device} model={tc['model']} train={len(train_ds)} val={len(val_ds)} classes={len(classes)}")

    teacher = None
    if args.teacher:
        teacher, tmeta = load_checkpoint(args.teacher, device)
        if tmeta["classes"] != classes:
            raise SystemExit("Teacher was trained on a different class list.")
        if (tmeta["mean"], tmeta["std"]) != (mean, std):
            print("NOTE: teacher and student use different normalisation; the teacher sees the student's inputs.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total_steps = tc["epochs"] * len(train_loader)
    warmup = max(1, min(500, total_steps // 10))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = bool(tc["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=tc["label_smoothing"])

    meta = {
        "model_name": tc["model"],
        "classes": classes,
        "class_info": class_info,
        "image_size": size,
        "mean": mean,
        "std": std,
        "drop_rate": tc["drop_rate"],
        "temperature": 1.0,
        "train_manifests": args.train_manifests,
        "augment": tc["augment"],
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "cli": vars(args)}), encoding="utf-8")
    log_f = open(run_dir / "log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["epoch", "train_loss", "train_acc", "val_acc", "lr", "seconds"])

    best_acc, bad_epochs = -1.0, 0
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        t0 = time.time()
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x)
                    loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd_loss(logits.float(), t_logits.float(), args.kd_temp)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)

        val_logits, val_labels = predict_logits(model, val_loader, device, desc="val")
        val_acc = (val_logits.argmax(1) == val_labels).float().mean().item()
        secs = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        log.writerow([epoch, f"{loss_sum / seen:.4f}", f"{correct / seen:.4f}", f"{val_acc:.4f}", f"{lr_now:.2e}", f"{secs:.0f}"])
        log_f.flush()
        print(f"epoch {epoch:2d} loss {loss_sum / seen:.4f} train_acc {correct / seen:.4f} val_acc {val_acc:.4f} ({secs:.0f}s)")

        if val_acc > best_acc:
            best_acc, bad_epochs = val_acc, 0
            save_checkpoint(run_dir / "best.pt", model, {**meta, "epoch": epoch, "val_acc": val_acc})
        else:
            bad_epochs += 1
            if bad_epochs >= tc["patience"]:
                print(f"Early stopping (no improvement for {bad_epochs} epochs).")
                break

    log_f.close()
    print(f"Best lab val accuracy {best_acc:.4f}. Checkpoint: {run_dir / 'best.pt'}")
    print(f"Next: python -m agrisight.evaluate --ckpt {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
