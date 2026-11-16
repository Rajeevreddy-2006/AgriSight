"""Build the assistant from config, skipping any component that is not set up yet."""
from __future__ import annotations

import os
import warnings
from pathlib import Path

from .agent.agent import AgriSight
from .llm import LLMClient

DEFAULT_CKPT = "runs/convnext_field/best.pt"


def build_agent(cfg: dict, ckpt: str | None = None) -> AgriSight:
    llm = LLMClient(**cfg["llm"])

    retriever = None
    index_dir = Path(cfg["paths"]["index_dir"])
    if (index_dir / "chunks.jsonl").exists():
        from .rag.retriever import HybridRetriever

        rc = cfg["rag"]
        retriever = HybridRetriever(index_dir, reranker=rc.get("reranker"), candidates=rc["candidates"], default_mode=rc["mode"])
    else:
        warnings.warn(f"No RAG index at {index_dir}; run `python -m agrisight.rag.index`.")

    predictor = None
    ckpt = ckpt or os.environ.get("AGRISIGHT_CKPT", DEFAULT_CKPT)
    if Path(ckpt).exists():
        from .predict import Predictor

        predictor = Predictor(ckpt, abstain_threshold=cfg["uncertainty"]["abstain_threshold"])
    else:
        warnings.warn(f"No classifier checkpoint at {ckpt}; set AGRISIGHT_CKPT. The VLM will be used alone.")

    vlm = None
    vc = cfg.get("vlm", {})
    if vc.get("backend", "none") != "none":
        from .vlm import VLM

        vlm = VLM(backend=vc["backend"], ollama_url=vc["ollama_url"], ollama_model=vc["ollama_model"], hf_model=vc["hf_model"])

    return AgriSight(llm, retriever=retriever, predictor=predictor, vlm=vlm,
                      top_k=cfg["rag"]["top_k"], retrieval_mode=cfg["rag"]["mode"],
                      use_weather=cfg.get("weather", {}).get("enabled", True))
