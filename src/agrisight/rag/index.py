"""Build the retrieval index from the documents in knowledge_base/.

  python -m agrisight.rag.index
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import load_config
from .chunking import chunk_documents, load_documents


def build_index(cfg: dict) -> Path:
    from sentence_transformers import SentenceTransformer

    rc = cfg["rag"]
    kb_dir, index_dir = Path(cfg["paths"]["kb_dir"]), Path(cfg["paths"]["index_dir"])
    docs = load_documents(kb_dir)
    if not docs:
        raise SystemExit(f"No .md/.txt/.pdf documents found in {kb_dir}")
    chunks = chunk_documents(docs, rc["chunk_size"], rc["chunk_overlap"])
    model = SentenceTransformer(rc["embed_model"])
    emb = model.encode([c["text"] for c in chunks], batch_size=32, normalize_embeddings=True, show_progress_bar=True)

    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", emb.astype(np.float32))
    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (index_dir / "meta.json").write_text(json.dumps({"embed_model": rc["embed_model"], "n_chunks": len(chunks)}), encoding="utf-8")
    print(f"Indexed {len(docs)} documents/pages into {len(chunks)} chunks -> {index_dir}")
    return index_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    build_index(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()
