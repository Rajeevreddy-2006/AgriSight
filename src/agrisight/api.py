"""REST API.

  uvicorn agrisight.api:app --reload --port 8000     (docs at http://localhost:8000/docs)
"""
from __future__ import annotations

import io
import os
import tempfile
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from .config import load_config
from .factory import build_agent

app = FastAPI(title="AgriSight API", version="0.1.0")


@lru_cache(maxsize=1)
def get_cfg():
    return load_config()


@lru_cache(maxsize=1)
def get_agent():
    return build_agent(get_cfg())


@lru_cache(maxsize=1)
def get_asr():
    from .speech import ASR

    return ASR(get_cfg()["asr"]["model"])


def _read_image(upload: UploadFile) -> Image.Image:
    try:
        return Image.open(io.BytesIO(upload.file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")


def _transcribe(upload: UploadFile, language: str) -> str:
    suffix = os.path.splitext(upload.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(upload.file.read())
        path = tmp.name
    try:
        return get_asr().transcribe(path, language)
    finally:
        os.unlink(path)


@app.get("/health")
def health():
    agent = get_agent()
    return {
        "status": "ok",
        "classifier": agent.predictor is not None,
        "retriever": agent.retriever is not None,
        "vlm": agent.vlm is not None,
    }


@app.post("/predict")
def predict(image: UploadFile = File(...)):
    agent = get_agent()
    if agent.predictor is None:
        raise HTTPException(503, "Classifier checkpoint not loaded (set AGRISIGHT_CKPT).")
    return agent.predictor.predict(_read_image(image))


@app.post("/transcribe")
def transcribe(audio: UploadFile = File(...), language: str = Form("en")):
    return {"text": _transcribe(audio, language)}


@app.post("/ask")
def ask(
    question: str = Form(""),
    language: str = Form("en"),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    use_rag: bool = Form(True),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
):
    transcript = None
    if audio is not None:
        transcript = _transcribe(audio, language)
        question = f"{question} {transcript}".strip()
    img = _read_image(image) if image is not None else None
    if not question and img is None:
        raise HTTPException(400, "Send a question, an audio clip or an image.")
    result = get_agent().ask(question, image=img, language=language, lat=lat, lon=lon, use_rag=use_rag)
    result["transcript"] = transcript
    return result
