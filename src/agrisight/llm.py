"""Minimal client for any OpenAI-compatible chat endpoint (Ollama, vLLM, OpenAI, Groq, ...)."""
from __future__ import annotations

import os

import requests


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key_env: str = "LLM_API_KEY",
                 temperature: float = 0.2, max_tokens: int = 700, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = os.environ.get(api_key_env, "not-needed")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def chat(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None) -> str:
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature if temperature is None else temperature,
                "max_tokens": max_tokens or self.max_tokens,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
