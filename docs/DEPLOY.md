# Deploying a public demo

A public link in your resume is worth more than a screenshot. The free option is
**Hugging Face Spaces** (Streamlit, CPU). A 7B LLM cannot run on a free CPU Space,
so the demo calls a hosted OpenAI-compatible API for the text model, while the image
classifier runs inside the Space.

## 1. Prepare
1. Finish training and pick the best model (`python scripts/make_report.py --print-best`).
   Use the distilled MobileNetV3 if its field accuracy is close: it is small and fast on CPU.
2. Get a free API key from a provider with an OpenAI-compatible endpoint
   (for example Groq: `https://api.groq.com/openai/v1`).

## 2. Create the Space
1. huggingface.co → New Space → SDK **Streamlit** → hardware **CPU basic**.
2. Push this repository to the Space (or connect it to the GitHub repo).
3. Upload the checkpoint to the Space (Git LFS) as `runs/convnext_field/best.pt`,
   or to a model repo and download it at start-up.
4. Upload `data/index/` built by `python -m agrisight.rag.index`.

## 3. Settings in the Space
Settings → Variables and secrets:

| Name | Value |
|---|---|
| `LLM_API_KEY` (secret) | your API key |
| `AGRISIGHT_CKPT` | `runs/convnext_field/best.pt` |
| `AGRISIGHT_CONFIG` | `configs/space.yaml` |

Create `configs/space.yaml` as a copy of `default.yaml` with:

```yaml
llm:
  base_url: https://api.groq.com/openai/v1
  model: <a model name your provider offers>
  api_key_env: LLM_API_KEY
vlm:
  backend: none        # no vision-language model on a free CPU Space
rag:
  reranker: null       # saves memory and time on CPU
```

Set the Space's app file to `app/streamlit_app.py`.

## Docker (any server)
```bash
docker build -t agrisight .
docker run -p 8000:8000 -e LLM_API_KEY=... -e AGRISIGHT_CKPT=/app/runs/convnext_field/best.pt \
  -v $(pwd)/runs:/app/runs -v $(pwd)/data/index:/app/data/index agrisight
```
