"""AgriSight assistant.

Pipeline (each step is recorded in `trace`):
  1. image classifier  -> disease + calibrated confidence
  2. VLM               -> symptom description (always if available; key when classifier is unsure)
  3. translate         -> question to English for retrieval (if needed)
  4. retrieval         -> official guideline passages
  5. weather           -> 3-day forecast for spray timing (if location given)
  6. LLM               -> grounded answer in the farmer's language, with citations
  7. grounding check   -> numbers not in evidence trigger one rewrite, then a warning

The order is fixed rather than left to the LLM: it is more reliable with small
local models and makes every step measurable.
"""
from __future__ import annotations

import time

from PIL import Image

from .grounding import unsupported_numbers
from .tools import get_weather

LANGUAGES = {
    "en": "English", "hi": "Hindi", "te": "Telugu", "ta": "Tamil", "kn": "Kannada",
    "mr": "Marathi", "bn": "Bengali", "or": "Odia", "gu": "Gujarati", "pa": "Punjabi",
}

SYSTEM_RAG = """You are AgriSight, an assistant for small farmers in India.
Rules:
1. Base every treatment or prevention step ONLY on the numbered SOURCES and cite them like [1].
2. Never write a chemical name, dose, concentration, interval or any other number that does not appear in the SOURCES, the question or the weather data. If the sources do not cover something, say so and suggest contacting the local Krishi Vigyan Kendra (KVK) or agriculture officer.
3. If the diagnosis is uncertain, say so clearly and ask for a closer, well-lit photo of the affected leaf.
4. Use simple words and short sentences, at most about 150 words. Answer in {language}."""

SYSTEM_NO_RAG = """You are AgriSight, an assistant for small farmers in India.
Answer the farmer's question helpfully in simple words, at most about 150 words. Answer in {language}."""


class AgriSight:
    def __init__(self, llm, retriever=None, predictor=None, vlm=None, top_k: int = 5,
                 retrieval_mode: str | None = None, use_weather: bool = True):
        self.llm = llm
        self.retriever = retriever
        self.predictor = predictor
        self.vlm = vlm
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode
        self.use_weather = use_weather

    def _translate_to_english(self, text: str, language: str) -> str:
        if language == "en" or not text.strip():
            return text
        return self.llm.chat(
            [{"role": "system", "content": "Translate the user's text to English. Output only the translation."},
             {"role": "user", "content": text}],
            temperature=0.0, max_tokens=200,
        )

    def ask(self, question: str = "", image: Image.Image | None = None, language: str = "en",
            lat: float | None = None, lon: float | None = None, use_rag: bool = True) -> dict:
        trace: list[dict] = []
        lang_name = LANGUAGES.get(language, "English")
        question = question.strip() or "What is wrong with my plant and what should I do?"

        def step(name, fn):
            t0 = time.time()
            try:
                out = fn()
                trace.append({"step": name, "ok": True, "seconds": round(time.time() - t0, 2)})
                return out
            except Exception as e:  # keep going: a missing component should not break the answer
                trace.append({"step": name, "ok": False, "error": str(e)[:300], "seconds": round(time.time() - t0, 2)})
                return None

        diagnosis = step("classifier", lambda: self.predictor.predict(image)) if (image is not None and self.predictor) else None
        vlm_text = step("vlm_describe", lambda: self.vlm.describe(image)) if (image is not None and self.vlm) else None
        question_en = step("translate", lambda: self._translate_to_english(question, language)) or question

        sources: list[dict] = []
        if use_rag and self.retriever is not None:
            topic = ""
            if diagnosis and not diagnosis["abstain"] and not diagnosis.get("healthy"):
                topic = f"{diagnosis['crop']} {diagnosis['disease']} management. "
            query = topic + question_en
            sources = step("retrieve", lambda: self.retriever.search(query, k=self.top_k, mode=self.retrieval_mode)) or []

        weather = None
        if self.use_weather and lat is not None and lon is not None:
            weather = step("weather", lambda: get_weather(lat, lon))

        # ---- build the prompt ------------------------------------------
        parts = []
        if diagnosis:
            certainty = "UNCERTAIN" if diagnosis["abstain"] else "fairly confident"
            alts = ", ".join(f"{t['display']} ({t['prob']:.0%})" for t in diagnosis["top_k"][1:])
            parts.append(f"IMAGE MODEL DIAGNOSIS: {diagnosis['display']} (confidence {diagnosis['confidence']:.0%}, {certainty}). "
                         f"Other possibilities: {alts}.")
        elif image is not None:
            parts.append("IMAGE MODEL DIAGNOSIS: not available.")
        if vlm_text:
            parts.append(f"VISUAL DESCRIPTION (vision-language model, may be wrong): {vlm_text}")
        if weather:
            parts.append(f"WEATHER FORECAST: {weather['summary']}")
        if use_rag:
            if sources:
                src_txt = "\n\n".join(
                    f"[{i + 1}] ({s['source']}{', p.' + str(s['page']) if s.get('page') else ''})\n{s['text']}"
                    for i, s in enumerate(sources)
                )
                parts.append(f"SOURCES:\n{src_txt}")
            else:
                parts.append("SOURCES: none found. Do not give treatment details; refer the farmer to the local KVK.")
        parts.append(f"FARMER'S QUESTION: {question}")
        user_msg = "\n\n".join(parts)

        system = (SYSTEM_RAG if use_rag else SYSTEM_NO_RAG).format(language=lang_name)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
        answer = step("llm_answer", lambda: self.llm.chat(messages))
        if answer is None:
            answer = "Sorry, the assistant is not available right now. Please contact your local Krishi Vigyan Kendra (KVK)."

        # ---- grounding check -------------------------------------------
        evidence = "\n".join([question, question_en, user_msg])
        unsupported = unsupported_numbers(answer, evidence) if use_rag else []
        if unsupported:
            fix = (f"Your answer contains numbers that are not in the sources: {', '.join(unsupported)}. "
                   "Rewrite the answer without any number that is not in the sources. Keep the citations.")
            retry = step("llm_rewrite", lambda: self.llm.chat(messages + [{"role": "assistant", "content": answer},
                                                                         {"role": "user", "content": fix}]))
            if retry:
                answer = retry
                unsupported = unsupported_numbers(answer, evidence)

        return {
            "answer": answer,
            "language": language,
            "question_en": question_en,
            "diagnosis": diagnosis,
            "vlm_description": vlm_text,
            "sources": [{"n": i + 1, "source": s["source"], "page": s.get("page"), "text": s["text"]} for i, s in enumerate(sources)],
            "weather": weather,
            "unsupported_numbers": unsupported,
            "trace": trace,
        }
