"""Vision-language model wrapper (Ollama or Hugging Face transformers).

Used to (1) describe visible symptoms in words and (2) act as a second opinion
when the image classifier is unsure or the crop is outside its classes.
"""
from __future__ import annotations

import base64
import io
import re

import requests
from PIL import Image

DESCRIBE_PROMPT = (
    "You are a plant pathologist. Look at this photo of a plant leaf. "
    "Describe the visible symptoms briefly: the crop if you can recognise it, colour and shape of any "
    "spots or lesions, where they are on the leaf, and any mould, curling, holes or insects. "
    "Then give the most likely problem and one or two alternatives, and say how confident you are. "
    "If the photo is not a plant or is too blurry to judge, say so. Keep it under 120 words."
)


def _to_b64(image: Image.Image, max_side: int = 1024) -> str:
    img = image.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


class VLM:
    def __init__(self, backend: str = "ollama", ollama_url: str = "http://localhost:11434",
                 ollama_model: str = "qwen2.5vl:7b", hf_model: str = "Qwen/Qwen2.5-VL-3B-Instruct", timeout: int = 180):
        self.backend = backend
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model
        self.hf_model = hf_model
        self.timeout = timeout
        self._hf = None

    # ---- backends -------------------------------------------------------
    def _ollama(self, image: Image.Image, prompt: str, max_tokens: int) -> str:
        r = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.ollama_model,
                "messages": [{"role": "user", "content": prompt, "images": [_to_b64(image)]}],
                "stream": False,
                "options": {"temperature": 0, "num_predict": max_tokens},
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def _load_hf(self):
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:  # older transformers
            from transformers import AutoModelForVision2Seq as AutoVLM
        processor = AutoProcessor.from_pretrained(self.hf_model)
        model = AutoVLM.from_pretrained(self.hf_model, torch_dtype="auto", device_map="auto")
        self._hf = (processor, model)

    def _hf_generate(self, image: Image.Image, prompt: str, max_tokens: int) -> str:
        if self._hf is None:
            self._load_hf()
        processor, model = self._hf
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image.convert("RGB")], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    # ---- public API -----------------------------------------------------
    def generate(self, image: Image.Image, prompt: str, max_tokens: int = 300) -> str:
        if self.backend == "ollama":
            return self._ollama(image, prompt, max_tokens)
        if self.backend == "hf":
            return self._hf_generate(image, prompt, max_tokens)
        raise ValueError(f"VLM backend '{self.backend}' is disabled or unknown")

    def describe(self, image: Image.Image) -> str:
        return self.generate(image, DESCRIBE_PROMPT)

    def classify(self, image: Image.Image, options: list[str]) -> int | None:
        """Zero-shot: pick one of `options`. Returns its index, or None if unparseable."""
        numbered = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options))
        prompt = (
            "Which ONE of these best describes the plant leaf in the photo?\n"
            f"{numbered}\n\nReply with only the number."
        )
        reply = self.generate(image, prompt, max_tokens=10)
        m = re.search(r"\d+", reply)
        if not m:
            return None
        idx = int(m.group()) - 1
        return idx if 0 <= idx < len(options) else None
