# 8-week plan

About 10–12 hours a week. Free Colab is enough for weeks 1–4 with ResNet-50 /
ConvNeXt-Tiny at 224 px; Colab Pro or a college GPU makes weeks 4–6 faster.
Run Ollama on your laptop (or use the Hugging Face backend on Colab) for the LLM/VLM.

## Week 1 — Data and baseline
- [ ] Download PlantVillage and PlantDoc, run `prepare_data.py`, check the per-class counts
- [ ] Train ResNet-50 (`--augment basic`) and evaluate on lab and field
- [ ] Write down the lab → field gap. This is your opening result.

## Week 2 — Closing the gap
- [ ] Field-style augmentation; ConvNeXt-T and ViT-S backbones
- [ ] Look at the field confusion matrix and 30 wrong images
- [ ] Fill the RQ1 table for these runs

## Week 3 — Domain adaptation and uncertainty
- [ ] AdaBN on ResNet-50, DANN on the best model
- [ ] Temperature scaling, abstention threshold, risk–coverage curve
- [ ] Oracle run (trained with field labels) to show the upper bound

## Week 4 — Generative AI layer
- [ ] Install Ollama, pull `qwen2.5:7b-instruct` and `qwen2.5vl:7b`
- [ ] Collect 15–30 official guideline PDFs into `knowledge_base/`, build the index
- [ ] Run the Streamlit app end to end
- [ ] Classifier vs VLM vs hybrid experiment (RQ3)

## Week 5 — Evaluation of the assistant
- [ ] Write 50+ evaluation questions (some in Telugu/Hindi) with reference keywords and expected source
- [ ] Run `rag_eval.py`: LLM-only vs RAG, and the four retrieval modes (RQ4)
- [ ] Read 20 answers yourself and note failure types

## Week 6 — Voice and languages
- [ ] Whisper ASR in the app; test 20 voice questions in your language
- [ ] Optional stretch: LoRA fine-tune Whisper-small on an AI4Bharat dataset and compare word error rate

## Week 7 — Deployment
- [ ] Distil into MobileNetV3, quantize, measure size and latency (RQ5)
- [ ] Deploy the demo (Hugging Face Spaces with the HF backend, or Docker)
- [ ] Record a 2-minute demo video

## Week 8 — Write-up
- [ ] 4–6 page report in paper format: problem, related work, method, experiments, error analysis, limitations, ethics
- [ ] Clean README with the results tables and the demo GIF
- [ ] Put the report on arXiv-style PDF in `docs/` and link it from the README

## Resume bullets (fill in your real numbers)

- Built a multimodal crop-disease assistant (CNN/ViT + vision-language model + RAG + Whisper) answering farmers in Telugu/Hindi with cited, guideline-grounded advice.
- Measured a **X-point** accuracy drop from lab (PlantVillage) to field (PlantDoc) photos and recovered **Y points** with field augmentation and domain adaptation (DANN/AdaBN).
- Reduced answers with unsupported dosage numbers from **A%** to **B%** using hybrid retrieval, reranking and a numeric grounding check.
- Distilled and int8-quantized the model to **N MB / M ms** on a single CPU thread for offline use on low-cost phones.

## Interview questions you should be able to answer

- Why does a 99% PlantVillage model fail on field photos? (shortcut learning on backgrounds, dataset bias)
- How does DANN's gradient reversal work, and why does AdaBN only help BatchNorm models?
- What is ECE? Why does temperature scaling not change accuracy?
- Why hybrid BM25 + dense retrieval? What does the cross-encoder add, and what does it cost?
- How do you stop an LLM from inventing pesticide doses, and how did you measure it?
- What would you change to serve 1 million farmers? (caching, smaller models, on-device classifier, async queue)
