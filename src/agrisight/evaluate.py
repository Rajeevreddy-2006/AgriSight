"""Evaluate a checkpoint on lab and field test sets: accuracy, macro-F1, calibration,
selective prediction (abstaining when unsure), and the lab -> field gap.

Examples:
  python -m agrisight.evaluate --ckpt runs/convnext_field/best.pt
  # Test-time adaptation with AdaBN (needs a BatchNorm model such as resnet50):
  python -m agrisight.evaluate --ckpt runs/resnet50/best.pt --adabn-manifest field_train
  # Store the fitted temperature in the checkpoint for the app to use:
  python -m agrisight.evaluate --ckpt runs/convnext_field/best.pt --save-temperature
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from . import metrics as M
from .config import get_device, load_config, manifest_path
from .data import ManifestDataset, build_transforms
from .domain_adapt import adabn
from .engine import make_loader, predict_logits
from .models import load_checkpoint
from .uncertainty import fit_temperature


def summarize(probs: np.ndarray, probs_cal: np.ndarray, labels: np.ndarray, classes: list[str], threshold: float) -> dict:
    preds = probs.argmax(1)
    present = np.unique(labels)
    per_class = {}
    for c in present:
        m = labels == c
        per_class[classes[c]] = {"n": int(m.sum()), "accuracy": float((preds[m] == c).mean())}
    _, _, aurc = M.risk_coverage(probs_cal, labels)
    return {
        "n": int(len(labels)),
        "accuracy": M.accuracy(probs, labels),
        "macro_f1": float(f1_score(labels, preds, labels=present, average="macro", zero_division=0)),
        "nll": M.nll(probs, labels),
        "ece": M.ece(probs, labels),
        "ece_calibrated": M.ece(probs_cal, labels),
        "aurc": aurc,
        "selective": M.selective(probs_cal, labels, threshold),
        "per_class": per_class,
    }


def plot_confusion(labels, preds, classes, path: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))), normalize="true")
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(np.nan_to_num(cm), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--manifests", default="lab_test,field_test")
    ap.add_argument("--calib-manifest", default="lab_val", help="held-out set for temperature scaling")
    ap.add_argument("--adabn-manifest", help="unlabelled field images for AdaBN, e.g. field_train")
    ap.add_argument("--save-temperature", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. adabn")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.ckpt, device)
    classes = ckpt["classes"]
    tf = build_transforms(ckpt["image_size"], ckpt["mean"], ckpt["std"], train=False)
    nw = cfg["data"]["num_workers"]
    threshold = cfg["uncertainty"]["abstain_threshold"]
    out_dir = Path(args.ckpt).parent
    tag = f"_{args.tag}" if args.tag else ""

    def loader_for(name):
        ds = ManifestDataset(manifest_path(cfg, name), classes, tf)
        return make_loader(ds, args.batch_size, False, nw, device)

    temperature = 1.0
    if args.calib_manifest:
        logits, labels = predict_logits(model, loader_for(args.calib_manifest), device, desc="calibration")
        temperature = fit_temperature(logits, labels)
        print(f"Fitted temperature on {args.calib_manifest}: T = {temperature:.3f}")

    results = {
        "checkpoint": str(args.ckpt),
        "run": Path(args.ckpt).parent.name,
        "model": ckpt["model_name"],
        "augment": ckpt.get("augment"),
        "train_manifests": ckpt.get("train_manifests", "lab_train"),
        "adaptation": "adabn" if args.adabn_manifest else ckpt.get("adaptation", "none"),
        "temperature": temperature,
        "abstain_threshold": threshold,
        "sets": {},
    }
    for name in args.manifests.split(","):
        m = model
        if args.adabn_manifest and name.startswith("field"):
            m = adabn(copy.deepcopy(model), loader_for(args.adabn_manifest), device)
        logits, labels = predict_logits(m, loader_for(name), device, desc=name)
        logits_np, labels_np = logits.numpy(), labels.numpy()
        probs = M.softmax(logits_np)
        probs_cal = M.softmax(logits_np / temperature)
        results["sets"][name] = summarize(probs, probs_cal, labels_np, classes, threshold)
        plot_confusion(labels_np, probs.argmax(1), classes, out_dir / f"confusion_{name}{tag}.png", f"{ckpt['model_name']} on {name}")

    sets = results["sets"]
    if "lab_test" in sets and "field_test" in sets:
        results["lab_to_field_gap"] = sets["lab_test"]["accuracy"] - sets["field_test"]["accuracy"]

    out_path = out_dir / f"eval{tag}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n| set | n | accuracy | macro-F1 | ECE | ECE (calibrated) | coverage @{threshold} | acc on kept |")
    print("|---|---|---|---|---|---|---|---|")
    for name, r in sets.items():
        s = r["selective"]
        print(f"| {name} | {r['n']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['ece']:.3f} | "
              f"{r['ece_calibrated']:.3f} | {s['coverage']:.3f} | {s['accuracy_on_kept']:.3f} |")
    if "lab_to_field_gap" in results:
        print(f"\nLab -> field accuracy gap: {results['lab_to_field_gap'] * 100:.1f} points")
    print(f"Saved {out_path}")

    if args.save_temperature:
        raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        raw["temperature"] = temperature
        torch.save(raw, args.ckpt)
        print(f"Stored T={temperature:.3f} in {args.ckpt}")


if __name__ == "__main__":
    main()
