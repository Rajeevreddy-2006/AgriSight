"""Speech in (Whisper ASR) and speech out (optional gTTS) for Indian languages.

For better Indian-language accuracy, swap the ASR model for a fine-tuned one
(e.g. AI4Bharat IndicWhisper checkpoints, or your own LoRA fine-tune).
"""
from __future__ import annotations

import io

WHISPER_LANG = {
    "en": "english", "hi": "hindi", "te": "telugu", "ta": "tamil", "kn": "kannada",
    "mr": "marathi", "bn": "bengali", "gu": "gujarati", "pa": "punjabi",
}


class ASR:
    def __init__(self, model: str = "openai/whisper-small", device: int | None = None):
        import torch
        from transformers import pipeline

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("automatic-speech-recognition", model=model, device=device)

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        kwargs = {"task": "transcribe"}
        if language in WHISPER_LANG:
            kwargs["language"] = WHISPER_LANG[language]
        out = self.pipe(audio_path, generate_kwargs=kwargs, chunk_length_s=30)
        return out["text"].strip()


def text_to_speech(text: str, language: str = "en") -> bytes | None:
    """MP3 bytes via gTTS (needs internet). Returns None if unavailable."""
    try:
        from gtts import gTTS

        buf = io.BytesIO()
        gTTS(text=text, lang=language).write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None
