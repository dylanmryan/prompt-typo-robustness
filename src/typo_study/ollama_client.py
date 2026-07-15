"""Thin HTTP client for a local Ollama server."""
from __future__ import annotations

import time

import requests


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 180,
        max_retries: int = 3,
        backoff_s: float = 2.0,
        temperature: float = 0.0,
        num_predict: int = 512,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.options = {"temperature": temperature, "num_predict": num_predict}

    def generate(self, model: str, prompt: str) -> str:
        payload = {"model": model, "prompt": prompt, "stream": False, "options": self.options}
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/generate", json=payload, timeout=self.timeout_s)
                resp.raise_for_status()
                return resp.json()["response"]
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as err:
                last_err = err
                time.sleep(self.backoff_s * (attempt + 1))
        raise OllamaError(f"generate failed after {self.max_retries} attempts: {last_err}")
