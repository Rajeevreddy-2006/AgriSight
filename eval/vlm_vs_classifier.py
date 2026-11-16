"""Research question 1: on real field photos, does a fine-tuned classifier, a zero-shot
vision-language model, or a confidence-based hybrid diagnose best?

Hybrid rule: trust the classifier when its calibrated confidence >= t, otherwise ask the VLM.

  python eval/vlm_vs_classifier.py --ckpt runs/convnext_field/best.pt --limit 300
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agrisight.config import load_config, manifest_path  # noqa: E402
from agrisight.data import read_manifest  # noqa: E402
from agrisight.predict import Predictor  # noqa: E402
from agrisight.vlm import VLM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--manifest", default="field_test")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--thresholds", default="0.3,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--out", default="runs/vlm_vs_classifier.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    vc = cfg["vlm"]
    predictor = Predictor(args.ckpt, tta=False)
    vlm = VLM(backend=vc["backend"], ollama_url=vc["ollama_url"], ollama_model=vc["ollama_model"], hf_model=vc["hf_model"])
    classes = predictor.classes
    options = [predictor.display(c) for c in classes]

    rows = read_manifest(manifest_path(cfg, args.manifest))
    random.Random(0).shuffle(rows)
    rows = [r for r in rows if r[1] in classes][: args.limit]

    y, clf_pred, clf_conf, vlm_pred = [], [], [], []
    for path, label in tqdm(rows, desc="images"):
        img = Image.open(path).convert("RGB")
        p = predictor.predict(img)
        try:
            v = vlm.classify(img, options)
        except Exception as e:
            print("VLM error:", e)
            v = None
        y.append(classes.index(label))
        clf_pred.append(classes.index(p["label"]))
        clf_conf.append(p["confidence"])
        vlm_pred.append(-1 if v is None else v)

    y, clf_pred, clf_conf, vlm_pred = map(np.array, (y, clf_pred, clf_conf, vlm_pred))
    res = {
        "n": int(len(y)),
        "classifier_accuracy": float((clf_pred == y).mean()),
        "vlm_accuracy": float((vlm_pred == y).mean()),
        "vlm_unparseable": float((vlm_pred == -1).mean()),
        "hybrid": [],
    }
    for t in map(float, args.thresholds.split(",")):
        use_vlm = (clf_conf < t) & (vlm_pred >= 0)
        pred = np.where(use_vlm, vlm_pred, clf_pred)
        res["hybrid"].append({"threshold": t, "accuracy": float((pred == y).mean()), "vlm_used": float(use_vlm.mean())})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nn={res['n']}  classifier {res['classifier_accuracy']:.3f}  VLM {res['vlm_accuracy']:.3f}")
    print("| threshold | hybrid accuracy | share sent to VLM |\n|---|---|---|")
    for h in res["hybrid"]:
        print(f"| {h['threshold']} | {h['accuracy']:.3f} | {h['vlm_used']:.2f} |")


if __name__ == "__main__":
    main()
