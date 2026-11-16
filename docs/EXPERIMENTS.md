# Experiments

**Run everything with one command:** `bash scripts/run_all.sh all` (or the Colab notebook
`notebooks/AgriSight_full_pipeline.ipynb`). The tables are generated automatically into
[RESULTS.md](RESULTS.md). This page explains each experiment and the individual commands.

Never copy numbers from papers into your results. If time allows, repeat the key runs with a
second seed (`seed:` in `configs/default.yaml`) and report mean ± std.

## RQ1. How large is the lab → field gap, and what closes it?

Train on PlantVillage only, test on PlantVillage (lab) and PlantDoc (field).

```bash
python -m agrisight.train --run-name resnet50_basic   --model resnet50 --augment basic
python -m agrisight.train --run-name resnet50_field   --model resnet50 --augment field
python -m agrisight.train --run-name convnext_basic   --model convnext_tiny.fb_in22k --augment basic
python -m agrisight.train --run-name convnext_field   --model convnext_tiny.fb_in22k --augment field
python -m agrisight.train --run-name vit_field        --model vit_small_patch16_224.augreg_in21k --augment field
python -m agrisight.evaluate --ckpt runs/<run>/best.pt
python -m agrisight.evaluate --ckpt runs/resnet50_field/best.pt --adabn-manifest field_train --tag adabn
python -m agrisight.domain_adapt --init runs/convnext_field/best.pt --run-name convnext_dann
python -m agrisight.train --run-name oracle --train-manifests lab_train,field_train   # upper bound
```

| Model | Augment | Adaptation | Lab acc | Field acc | Gap | Field macro-F1 |
|---|---|---|---|---|---|---|
| ResNet-50 | basic | none | | | | |
| ResNet-50 | field | none | | | | |
| ResNet-50 | field | AdaBN | | | | |
| ConvNeXt-T | basic | none | | | | |
| ConvNeXt-T | field | none | | | | |
| ConvNeXt-T | field | DANN | | | | |
| ViT-S | field | none | | | | |
| ConvNeXt-T | field | trained on field labels (oracle) | | | | |

Questions to answer in the write-up: Which change closes the gap most? Do
ImageNet-21k-pretrained backbones transfer better? Which classes collapse in the
field (look at `confusion_field_test.png`), and why?

## RQ2. Does the model know when it is wrong?

From `eval.json`: ECE before/after temperature scaling, AURC, and coverage vs
accuracy at the abstain threshold, for lab and field.

| Model | Field ECE | Field ECE (calibrated) | AURC | Coverage @0.6 | Accuracy on kept |
|---|---|---|---|---|---|
| | | | | | |

Key point to discuss: temperature is fitted on **lab** validation data. Does
calibration still hold on field data? (Usually not fully. Explain why.)

## RQ3. Classifier vs vision-language model vs hybrid on field photos

```bash
python eval/vlm_vs_classifier.py --ckpt runs/convnext_field/best.pt --limit 300
```

| Method | Field accuracy | Notes |
|---|---|---|
| Fine-tuned classifier | | |
| Zero-shot VLM (Qwen2.5-VL) | | |
| Hybrid, threshold = best | | share sent to VLM: |

## RQ4. Does retrieval reduce hallucinated advice?

```bash
python -m agrisight.rag.index
python eval/rag_eval.py --questions eval/questions.jsonl
```

| Setting | Answers with unsupported numbers | Keyword coverage | Citation rate |
|---|---|---|---|
| LLM only | | | |
| LLM + RAG + grounding check | | | |

| Retrieval | hit@5 |
|---|---|
| BM25 | |
| Dense (bge-small) | |
| Hybrid (RRF) | |
| Hybrid + reranker | |

Write at least 50 questions for the final version, with an agronomist or an official
document as the reference, and include questions in Telugu/Hindi.

## RQ5. Can it run on a cheap phone?

```bash
python -m agrisight.train --run-name student_mnv3 --model mobilenetv3_large_100 --teacher runs/convnext_field/best.pt
python -m agrisight.compress --ckpt runs/student_mnv3/best.pt --threads 1
```

| Model | Size (MB) | CPU latency, 1 thread (ms) | Field acc |
|---|---|---|---|
| ConvNeXt-T fp32 (teacher) | | | |
| MobileNetV3 fp32 (no distillation) | | | |
| MobileNetV3 fp32 (distilled) | | | |
| MobileNetV3 int8 (distilled) | | | |

## Error analysis (do not skip)

Pick 30 field images the best model gets wrong. For each, note the cause: background
clutter, several leaves, early-stage symptom, look-alike disease, label noise,
lighting. Put a grid of examples and a table of causes in the report.
