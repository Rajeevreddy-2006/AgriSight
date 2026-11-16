#!/usr/bin/env bash
# Runs every experiment in docs/EXPERIMENTS.md, then writes docs/RESULTS.md and the
# README results table. Finished runs are skipped, so after a Colab disconnect just
# run the same command again and it continues where it stopped.
#
# Usage:
#   bash scripts/run_all.sh vision      # RQ1, RQ2, RQ5 (needs a GPU)
#   bash scripts/run_all.sh assistant   # RQ3, RQ4 (needs Ollama or another LLM endpoint)
#   bash scripts/run_all.sh report      # only rebuild the results tables
#   bash scripts/run_all.sh all
#
# Environment:
#   SMOKE=1      tiny version of everything into runs_smoke/ (checks the pipeline in minutes)
#   EPOCHS=8     override the number of training epochs
#   VLM_LIMIT=300  images for the classifier-vs-VLM comparison
set -euo pipefail
STAGE=${1:-all}
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

TRAIN_EXTRA=()
DANN_EXTRA=()
VLM_LIMIT=${VLM_LIMIT:-300}
if [ "${SMOKE:-0}" = "1" ]; then
  export AGRISIGHT_RUNS_DIR=${AGRISIGHT_RUNS_DIR:-runs_smoke}
  TRAIN_EXTRA+=(--limit 256 --epochs 1)
  DANN_EXTRA+=(--epochs 1 --max-steps 5)
  VLM_LIMIT=10
elif [ -n "${EPOCHS:-}" ]; then
  TRAIN_EXTRA+=(--epochs "$EPOCHS")
fi
RUNS=$(python -c "from agrisight.config import load_config; print(load_config()['paths']['runs_dir'])")
mkdir -p "$RUNS"
echo "Outputs -> $RUNS"

train() {  # train <run-name> [args...]
  local name=$1; shift
  if [ -f "$RUNS/$name/best.pt" ]; then echo "[skip] train $name (done)"; return; fi
  echo "[run ] train $name"
  python -m agrisight.train --run-name "$name" "${TRAIN_EXTRA[@]}" "$@"
}

evaluate() {  # evaluate <run-name> <tag or -> [args...]
  local name=$1 tag=$2; shift 2
  local suffix=""; [ "$tag" != "-" ] && suffix="_$tag"
  if [ -f "$RUNS/$name/eval$suffix.json" ]; then echo "[skip] evaluate $name $tag (done)"; return; fi
  echo "[run ] evaluate $name $tag"
  local tag_args=(); [ "$tag" != "-" ] && tag_args=(--tag "$tag")
  python -m agrisight.evaluate --ckpt "$RUNS/$name/best.pt" "${tag_args[@]}" "$@"
}

compress() {
  local name=$1
  if [ -f "$RUNS/$name/compression.json" ]; then echo "[skip] compress $name (done)"; return; fi
  echo "[run ] compress $name"
  python -m agrisight.compress --ckpt "$RUNS/$name/best.pt" --threads 1
}

vision() {
  # RQ1: backbones x augmentation, trained on lab photos only
  train resnet50_basic --model resnet50 --augment basic
  train resnet50_field --model resnet50 --augment field
  train convnext_basic --model convnext_tiny.fb_in22k --augment basic
  train convnext_field --model convnext_tiny.fb_in22k --augment field
  train vit_field      --model vit_small_patch16_224.augreg_in21k --augment field
  # Upper bound: field labels used in training
  train oracle         --model convnext_tiny.fb_in22k --augment field --train-manifests lab_train,field_train
  for r in resnet50_basic resnet50_field convnext_basic convnext_field vit_field oracle; do
    evaluate "$r" - --save-temperature
  done
  # Domain adaptation without field labels
  evaluate resnet50_field adabn --adabn-manifest field_train
  if [ ! -f "$RUNS/convnext_dann/best.pt" ]; then
    echo "[run ] DANN"
    python -m agrisight.domain_adapt --init "$RUNS/convnext_field/best.pt" --run-name convnext_dann "${DANN_EXTRA[@]}"
  fi
  evaluate convnext_dann - --save-temperature
  # RQ5: small phone model, with and without distillation, then int8
  train mnv3_plain     --model mobilenetv3_large_100 --augment field
  train mnv3_distilled --model mobilenetv3_large_100 --augment field --teacher "$RUNS/convnext_field/best.pt"
  for r in mnv3_plain mnv3_distilled; do evaluate "$r" - --save-temperature; done
  compress convnext_field
  compress mnv3_distilled
}

assistant() {
  python -m agrisight.rag.index
  BEST=$(python scripts/make_report.py --print-best)
  echo "Best field model: $BEST"
  if [ ! -f "$RUNS/vlm_vs_classifier.json" ]; then
    python eval/vlm_vs_classifier.py --ckpt "$RUNS/$BEST/best.pt" --limit "$VLM_LIMIT" --out "$RUNS/vlm_vs_classifier.json"
  fi
  if [ ! -f "$RUNS/rag_eval.json" ]; then
    Q=eval/questions.jsonl; [ -f "$Q" ] || Q=eval/questions.sample.jsonl
    python eval/rag_eval.py --questions "$Q" --out "$RUNS/rag_eval.json"
  fi
}

case "$STAGE" in
  vision)    vision ;;
  assistant) assistant ;;
  report)    ;;
  all)       vision; assistant ;;
  *) echo "unknown stage: $STAGE"; exit 1 ;;
esac

if [ "${SMOKE:-0}" = "1" ]; then
  python scripts/make_report.py --out docs/RESULTS_smoke.md --no-readme
else
  python scripts/make_report.py
fi
