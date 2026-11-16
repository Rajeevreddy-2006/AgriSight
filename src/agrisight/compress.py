"""Export to ONNX, quantize to int8, and compare size / CPU latency / accuracy.

Example:
  python -m agrisight.compress --ckpt runs/student_mnv3/best.pt --limit 500

Note: dynamic int8 quantization helps transformer and linear-heavy models most.
For CNNs, static quantization with a calibration set (onnxruntime.quantization.
quantize_static) usually keeps accuracy better; add it as an experiment.
"""
from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import load_config, manifest_path
from .data import ManifestDataset, build_transforms
from .models import load_checkpoint


def export_onnx(model, size: int, path: Path):
    dummy = torch.randn(1, 3, size, size)
    kwargs = dict(
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(model, dummy, str(path), **kwargs)


def latency_ms(run, x: np.ndarray, warmup: int = 5, iters: int = 30) -> float:
    for _ in range(warmup):
        run(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        run(x)
    return (time.perf_counter() - t0) / iters * 1000


def main():
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--manifest", default="field_test")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--threads", type=int, default=1, help="CPU threads (1 approximates a cheap phone)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.set_num_threads(args.threads)
    model, meta = load_checkpoint(args.ckpt, "cpu")
    size = meta["image_size"]
    out_dir = Path(args.ckpt).parent
    fp32_path, int8_path = out_dir / "model_fp32.onnx", out_dir / "model_int8.onnx"

    export_onnx(model, size, fp32_path)
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    ds = ManifestDataset(manifest_path(cfg, args.manifest), meta["classes"],
                         build_transforms(size, meta["mean"], meta["std"], train=False), limit=args.limit)
    xs = np.stack([ds[i][0].numpy() for i in range(len(ds))])
    ys = np.array([ds.samples[i][1] for i in range(len(ds))])

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    variants = {}

    @torch.no_grad()
    def torch_run(x):
        return model(torch.from_numpy(x)).numpy()

    variants["pytorch_fp32"] = (torch_run, Path(args.ckpt).stat().st_size)
    for name, path in (("onnx_fp32", fp32_path), ("onnx_int8", int8_path)):
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        variants[name] = ((lambda s: lambda x: s.run(None, {"image": x})[0])(sess), path.stat().st_size)

    results = {}
    one = xs[:1]
    for name, (run, nbytes) in variants.items():
        preds = np.concatenate([run(xs[i : i + 32]).argmax(1) for i in range(0, len(xs), 32)])
        results[name] = {
            "size_mb": round(nbytes / 1e6, 2),
            "latency_ms_batch1": round(latency_ms(run, one), 2),
            "accuracy": float((preds == ys).mean()),
        }

    (out_dir / "compression.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n{meta['model_name']} on {args.manifest} (n={len(ys)}, threads={args.threads})")
    print("| variant | size (MB) | latency (ms) | accuracy |\n|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['size_mb']} | {r['latency_ms_batch1']} | {r['accuracy']:.3f} |")


if __name__ == "__main__":
    main()
