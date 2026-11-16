"""Collect every result JSON under runs/ into docs/RESULTS.md, a figure, and the README.

  python scripts/make_report.py                # write docs/RESULTS.md + update README
  python scripts/make_report.py --print-best   # print the run with the best field accuracy
                                               # (oracle excluded: it saw field labels)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agrisight.config import load_config  # noqa: E402

README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
PRETTY_MODEL = {
    "resnet50": "ResNet-50",
    "convnext_tiny.fb_in22k": "ConvNeXt-T (IN-22k)",
    "vit_small_patch16_224.augreg_in21k": "ViT-S/16 (IN-21k)",
    "mobilenetv3_large_100": "MobileNetV3-L",
    "efficientnet_b0": "EfficientNet-B0",
}


def pct(x) -> str:
    return "–" if x is None or x != x else f"{100 * x:.1f}"


def num(x, d=3) -> str:
    return "–" if x is None or x != x else f"{x:.{d}f}"


def load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def collect(runs: Path) -> dict:
    evals = []
    for p in sorted(runs.glob("*/eval*.json")):
        r = load_json(p)
        if not r or "sets" not in r:
            continue
        r.setdefault("run", p.parent.name)
        r["tag"] = p.stem[len("eval"):].lstrip("_") or None
        evals.append(r)
    comp = {p.parent.name: load_json(p) for p in sorted(runs.glob("*/compression.json"))}
    return {
        "evals": evals,
        "compression": {k: v for k, v in comp.items() if v},
        "vlm": load_json(runs / "vlm_vs_classifier.json"),
        "rag": load_json(runs / "rag_eval.json"),
    }


def label(r: dict) -> str:
    adapt = r.get("adaptation") or "none"
    train = r.get("train_manifests", "lab_train")
    extra = []
    if adapt != "none":
        extra.append(adapt.upper() if adapt in ("dann", "adabn") else adapt)
    if "field_train" in train:
        extra.append("field labels (oracle)")
    return ", ".join(extra) or "none"


def field_acc(r):
    return r["sets"].get("field_test", {}).get("accuracy")


def best_run(evals: list[dict]) -> dict | None:
    fair = [r for r in evals if "field_train" not in r.get("train_manifests", "") and field_acc(r) is not None]
    return max(fair, key=field_acc) if fair else None


def vision_table(evals):
    lines = ["| Run | Model | Augment | Adaptation | Lab acc | Field acc | Gap (pts) | Field macro-F1 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(evals, key=lambda r: (field_acc(r) is None, -(field_acc(r) or 0))):
        lab = r["sets"].get("lab_test", {}).get("accuracy")
        fld = r["sets"].get("field_test", {})
        gap = None if lab is None or not fld else lab - fld["accuracy"]
        run = r["run"] + (f" ({r['tag']})" if r.get("tag") else "")
        lines.append(f"| {run} | {PRETTY_MODEL.get(r['model'], r['model'])} | {r.get('augment') or '–'} | {label(r)} | "
                     f"{pct(lab)} | {pct(fld.get('accuracy'))} | {pct(gap)} | {pct(fld.get('macro_f1'))} |")
    return "\n".join(lines)


def calibration_table(evals):
    lines = ["| Run | Field ECE | Field ECE (temp. scaled) | AURC | Coverage @ threshold | Acc. on kept | Errors caught |",
             "|---|---|---|---|---|---|---|"]
    for r in evals:
        f = r["sets"].get("field_test")
        if not f:
            continue
        s = f["selective"]
        run = r["run"] + (f" ({r['tag']})" if r.get("tag") else "")
        lines.append(f"| {run} | {num(f['ece'])} | {num(f['ece_calibrated'])} | {num(f['aurc'])} | "
                     f"{pct(s['coverage'])}% @ {s['threshold']} | {pct(s['accuracy_on_kept'])} | {pct(s['errors_caught'])}% |")
    return "\n".join(lines)


def compression_table(comp, evals):
    lines = ["| Run | Variant | Size (MB) | CPU latency, 1 thread (ms) | Field acc (subset) |", "|---|---|---|---|---|"]
    for run, variants in comp.items():
        for v, m in variants.items():
            lines.append(f"| {run} | {v} | {m['size_mb']} | {m['latency_ms_batch1']} | {pct(m['accuracy'])} |")
    return "\n".join(lines)


def vlm_table(v):
    best = max(v["hybrid"], key=lambda h: h["accuracy"]) if v.get("hybrid") else None
    lines = ["| Method | Field accuracy | Notes |", "|---|---|---|",
             f"| Fine-tuned classifier | {pct(v['classifier_accuracy'])} | n = {v['n']} |",
             f"| Zero-shot VLM | {pct(v['vlm_accuracy'])} | unparseable replies: {pct(v['vlm_unparseable'])}% |"]
    if best:
        lines.append(f"| Hybrid (threshold {best['threshold']}) | {pct(best['accuracy'])} | {pct(best['vlm_used'])}% of images sent to the VLM |")
    return "\n".join(lines)


def rag_tables(rag):
    s = rag["summary"]
    lines = ["| Setting | Answers with unsupported numbers | Keyword coverage | Citation rate |", "|---|---|---|---|"]
    for name, title in (("no_rag", "LLM only"), ("rag", "LLM + RAG + grounding check")):
        m = s[name]
        lines.append(f"| {title} | {pct(m['unsupported_number_rate'])}% | {pct(m['keyword_coverage'])}% | {pct(m['citation_rate'])}% |")
    if s.get("retrieval_hit_at_k"):
        lines += ["", "| Retrieval | hit@k |", "|---|---|"]
        lines += [f"| {m} | {pct(v)}% |" for m, v in s["retrieval_hit_at_k"].items()]
    lines.append(f"\nQuestions evaluated: {len(rag.get('items', []))}")
    return "\n".join(lines)


def make_figure(evals, path: Path) -> bool:
    rows = [r for r in evals if "lab_test" in r["sets"] and "field_test" in r["sets"]]
    if not rows:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows.sort(key=lambda r: field_acc(r))
    names = [r["run"] + (f" ({r['tag']})" if r.get("tag") else "") for r in rows]
    lab = [100 * r["sets"]["lab_test"]["accuracy"] for r in rows]
    fld = [100 * r["sets"]["field_test"]["accuracy"] for r in rows]
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    c_lab, c_field = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (validated default palette)

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rows) + 1.6), facecolor=surface)
    ax.set_facecolor(surface)
    y = range(len(rows))
    h = 0.36
    ax.barh([i + h / 2 + 0.01 for i in y], lab, height=h, color=c_lab, label="Lab photos (PlantVillage)")
    ax.barh([i - h / 2 - 0.01 for i in y], fld, height=h, color=c_field, label="Field photos (PlantDoc)")
    for i, v in enumerate(fld):
        ax.text(v + 1, i - h / 2, f"{v:.1f}", va="center", fontsize=8, color=ink2)
    ax.set_yticks(list(y), names, fontsize=9, color=ink)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Test accuracy (%)", color=ink2, fontsize=9)
    ax.tick_params(colors=ink2, length=0)
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Accuracy on lab vs field photos", loc="left", fontsize=11, color=ink, pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5, labelcolor=ink2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)
    return True


def headline(data) -> str:
    evals = data["evals"]
    rows = ["| Experiment | Result |", "|---|---|"]
    base = next((r for r in evals if r["run"] == "resnet50_basic" and not r.get("tag")), None)
    best = best_run(evals)
    if base and "field_test" in base["sets"]:
        gap = base["sets"]["lab_test"]["accuracy"] - field_acc(base)
        rows.append(f"| Lab → field gap, ResNet-50 baseline | {pct(base['sets']['lab_test']['accuracy'])}% → "
                    f"{pct(field_acc(base))}% (−{pct(gap)} pts) |")
    if best:
        rows.append(f"| Best field accuracy without field labels | {pct(field_acc(best))}% ({best['run']}"
                    f"{', ' + best['tag'] if best.get('tag') else ''}) |")
        s = best["sets"]["field_test"]["selective"]
        rows.append(f"| Errors caught by abstaining (threshold {s['threshold']}) | {pct(s['errors_caught'])}% of field errors, "
                    f"{pct(s['coverage'])}% of photos still answered |")
    if data["vlm"]:
        v = data["vlm"]
        hb = max(v["hybrid"], key=lambda h: h["accuracy"])
        rows.append(f"| Classifier vs VLM vs hybrid (field) | {pct(v['classifier_accuracy'])}% / {pct(v['vlm_accuracy'])}% / {pct(hb['accuracy'])}% |")
    if data["rag"]:
        s = data["rag"]["summary"]
        rows.append(f"| Answers with unsupported numbers: LLM only → RAG | {pct(s['no_rag']['unsupported_number_rate'])}% → "
                    f"{pct(s['rag']['unsupported_number_rate'])}% |")
    phone = data["compression"].get("mnv3_distilled", {}).get("onnx_int8")
    if phone:
        rows.append(f"| Phone model (distilled MobileNetV3, int8) | {phone['size_mb']} MB, {phone['latency_ms_batch1']} ms on 1 CPU thread, "
                    f"{pct(phone['accuracy'])}% field acc |")
    if len(rows) == 2:
        return "_No results yet. Run `bash scripts/run_all.sh all`._"
    fig = "\n\n![Lab vs field accuracy](docs/figures/lab_vs_field.png)" if (ROOT / "docs/figures/lab_vs_field.png").exists() else ""
    return "\n".join(rows) + fig + "\n\nFull tables: [docs/RESULTS.md](docs/RESULTS.md)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="docs/RESULTS.md")
    ap.add_argument("--no-readme", action="store_true")
    ap.add_argument("--print-best", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    runs = Path(cfg["paths"]["runs_dir"])
    data = collect(runs)

    if args.print_best:
        best = best_run([r for r in data["evals"] if not r.get("tag")])
        if not best:
            sys.exit("No evaluated runs found.")
        print(best["run"])
        return

    out = ROOT / args.out
    fig_name = "lab_vs_field.png" if "smoke" not in args.out else "lab_vs_field_smoke.png"
    fig_path = ROOT / "docs" / "figures" / fig_name
    has_fig = make_figure(data["evals"], fig_path)

    parts = ["# Results", "", "_Generated by `scripts/make_report.py`. Do not edit by hand._", ""]
    if has_fig:
        parts += [f"![Lab vs field accuracy](figures/{fig_name})", ""]
    if data["evals"]:
        parts += ["## RQ1. Lab → field gap", "", vision_table(data["evals"]), "",
                  "## RQ2. Calibration and abstention (field test set)", "", calibration_table(data["evals"]), ""]
    if data["vlm"]:
        parts += ["## RQ3. Classifier vs vision-language model", "", vlm_table(data["vlm"]), ""]
    if data["rag"]:
        parts += ["## RQ4. Grounded advice", "", rag_tables(data["rag"]), ""]
    if data["compression"]:
        parts += ["## RQ5. Model size and speed", "", compression_table(data["compression"], data["evals"]), ""]
    if len(parts) <= 6:
        parts += ["_No results found in " + str(runs) + "._"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote {out}" + (f" and {fig_path}" if has_fig else ""))

    if not args.no_readme:
        readme = ROOT / "README.md"
        text = readme.read_text(encoding="utf-8")
        block = f"{README_START}\n{headline(data)}\n{README_END}"
        if README_START in text:
            text = re.sub(re.escape(README_START) + ".*?" + re.escape(README_END), lambda _: block, text, flags=re.S)
            readme.write_text(text, encoding="utf-8")
            print("Updated README results")


if __name__ == "__main__":
    main()
