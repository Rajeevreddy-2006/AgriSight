"""Research question 2: does grounding the LLM in official guidelines reduce hallucinated advice?

For every question it compares:
  * no_rag : the LLM answers alone
  * rag    : the full AgriSight pipeline with retrieved sources
and reports
  * unsupported-number rate : answers containing a number (dose, interval, %) absent
                              from the official reference passages
  * keyword coverage        : share of expected key points mentioned
  * citation rate           : answers citing at least one source (rag only)
It also reports retrieval hit@k for each retrieval mode when `expected_source` is set.

  python eval/rag_eval.py --questions eval/questions.sample.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agrisight.agent.agent import AgriSight  # noqa: E402
from agrisight.agent.grounding import unsupported_numbers  # noqa: E402
from agrisight.config import load_config  # noqa: E402
from agrisight.llm import LLMClient  # noqa: E402
from agrisight.rag.retriever import HybridRetriever  # noqa: E402


def keyword_coverage(answer: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    a = answer.lower()
    return sum(k.lower() in a for k in keywords) / len(keywords)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.sample.jsonl")
    ap.add_argument("--config", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--retrieval-modes", default="bm25,dense,hybrid,hybrid+rerank")
    ap.add_argument("--out", default="runs/rag_eval.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rc = cfg["rag"]
    retriever = HybridRetriever(cfg["paths"]["index_dir"], reranker=rc.get("reranker"), candidates=rc["candidates"])
    agent = AgriSight(LLMClient(**cfg["llm"]), retriever=retriever, top_k=args.k, retrieval_mode=rc["mode"], use_weather=False)
    items = [json.loads(l) for l in Path(args.questions).read_text(encoding="utf-8").splitlines() if l.strip()]

    modes = args.retrieval_modes.split(",")
    hits = {m: [] for m in modes}
    per_item = []
    for item in tqdm(items, desc="questions"):
        q = item["question"]
        if item.get("expected_source"):
            for m in modes:
                res = retriever.search(q, k=args.k, mode=m)
                hits[m].append(any(r["source"] == item["expected_source"] for r in res))

        reference = "\n".join(r["text"] for r in retriever.search(q, k=args.k))
        row = {"id": item.get("id"), "question": q}
        for name, use_rag in (("no_rag", False), ("rag", True)):
            out = agent.ask(q, language=item.get("language", "en"), use_rag=use_rag)
            evidence = q + "\n" + reference + "\n" + "\n".join(s["text"] for s in out["sources"])
            bad = unsupported_numbers(out["answer"], evidence)
            row[name] = {
                "answer": out["answer"],
                "unsupported_numbers": bad,
                "keyword_coverage": keyword_coverage(out["answer"], item.get("reference_keywords", [])),
                "cites": bool(re.search(r"\[\d+\]", out["answer"])),
            }
        per_item.append(row)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    summary = {}
    for name in ("no_rag", "rag"):
        summary[name] = {
            "unsupported_number_rate": mean([bool(r[name]["unsupported_numbers"]) for r in per_item]),
            "keyword_coverage": mean([r[name]["keyword_coverage"] for r in per_item]),
            "citation_rate": mean([r[name]["cites"] for r in per_item]),
        }
    summary["retrieval_hit_at_k"] = {m: mean(h) for m, h in hits.items() if h}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "items": per_item}, indent=2, ensure_ascii=False))

    fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    print(f"\nn={len(per_item)} questions")
    print("| setting | answers with unsupported numbers | keyword coverage | citation rate |\n|---|---|---|---|")
    for name in ("no_rag", "rag"):
        s = summary[name]
        print(f"| {name} | {fmt(s['unsupported_number_rate'])} | {fmt(s['keyword_coverage'])} | {fmt(s['citation_rate'])} |")
    if summary["retrieval_hit_at_k"]:
        print(f"\n| retrieval mode | hit@{args.k} |\n|---|---|")
        for m, v in summary["retrieval_hit_at_k"].items():
            print(f"| {m} | {fmt(v)} |")


if __name__ == "__main__":
    main()
