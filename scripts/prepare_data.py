"""Build train/val/test manifests for the lab (PlantVillage) and field (PlantDoc) domains.

Only the classes shared by both datasets (configs/label_map.yaml) are kept, so the
lab -> field comparison is apples to apples.

Outputs in --out (default data/manifests):
  lab_train.csv, lab_val.csv, lab_test.csv   stratified split of PlantVillage
  field_train.csv, field_test.csv            PlantDoc's own train/test split
  classes.json                               class list + crop/disease info

Example:
  python scripts/prepare_data.py \
      --plantvillage data/raw/PlantVillage-Dataset/raw/color \
      --plantdoc data/raw/PlantDoc-Dataset
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agrisight.labels import LabelMap, load_label_map  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan(root: Path, source: str, lm: LabelMap):
    per_class: dict[str, list[str]] = defaultdict(list)
    unmapped = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        label = lm.canonical(source, d.name)
        if label is None:
            unmapped.append(d.name)
            continue
        files = sorted(str(f.resolve()) for f in d.rglob("*") if f.suffix.lower() in IMG_EXT)
        per_class[label].extend(files)
    return per_class, unmapped


def write_csv(path: Path, rows: list[tuple[str, str]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plantvillage", required=True, type=Path, help="folder that contains the PlantVillage class folders")
    ap.add_argument("--plantdoc", required=True, type=Path, help="PlantDoc root containing train/ and test/")
    ap.add_argument("--label-map", default="configs/label_map.yaml")
    ap.add_argument("--out", default="data/manifests", type=Path)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lm = load_label_map(args.label_map)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Lab domain: stratified split per class.
    pv, pv_unmapped = scan(args.plantvillage, "plantvillage", lm)
    splits = {"lab_train": [], "lab_val": [], "lab_test": []}
    for label, files in sorted(pv.items()):
        files = files[:]
        rng.shuffle(files)
        n = len(files)
        n_test = max(1, int(n * args.test_fraction))
        n_val = max(1, int(n * args.val_fraction))
        splits["lab_test"] += [(f, label) for f in files[:n_test]]
        splits["lab_val"] += [(f, label) for f in files[n_test : n_test + n_val]]
        splits["lab_train"] += [(f, label) for f in files[n_test + n_val :]]

    # Field domain: keep PlantDoc's official split.
    pd_unmapped = []
    field_counts = {}
    for split in ("train", "test"):
        root = args.plantdoc / split
        if not root.exists():
            sys.exit(f"Missing {root}. --plantdoc must contain train/ and test/ folders.")
        per_class, unmapped = scan(root, "plantdoc", lm)
        pd_unmapped += unmapped
        rows = [(f, label) for label, files in sorted(per_class.items()) for f in files]
        splits[f"field_{split}"] = rows
        field_counts[split] = {k: len(v) for k, v in per_class.items()}

    for name, rows in splits.items():
        rng.shuffle(rows)
        write_csv(args.out / f"{name}.csv", rows)

    (args.out / "classes.json").write_text(
        json.dumps({"classes": lm.classes, "info": lm.info}, indent=2), encoding="utf-8"
    )

    # Summary
    print(f"{'class':34s} {'lab':>7s} {'field_tr':>9s} {'field_te':>9s}")
    for c in lm.classes:
        print(f"{c:34s} {len(pv.get(c, [])):7d} {field_counts['train'].get(c, 0):9d} {field_counts['test'].get(c, 0):9d}")
    for name, rows in splits.items():
        print(f"{name:12s} {len(rows):7d} images")
    missing = [c for c in lm.classes if c not in pv]
    if missing:
        print("\nWARNING: classes with no PlantVillage images:", missing)
    if pv_unmapped:
        print(f"\nIgnored {len(pv_unmapped)} PlantVillage folders not shared with PlantDoc (expected).")
    if pd_unmapped:
        print("\nWARNING: PlantDoc folders not in label_map.yaml (add them if they should be used):", sorted(set(pd_unmapped)))
    print(f"\nManifests written to {args.out}/")


if __name__ == "__main__":
    main()
