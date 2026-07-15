"""Retry behavior tests for the thin Ollama HTTP client."""
import pytest
import requests

from typo_study.ollama_client import OllamaClient, OllamaError


class _FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"response": "hello"}


def test_generate_returns_response_text(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse())
    client = OllamaClient(backoff_s=0)
    assert client.generate("m", "p") == "hello"


def test_generate_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.ConnectionError("refused")
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", flaky)
    client = OllamaClient(max_retries=3, backoff_s=0)
    assert client.generate("m", "p") == "hello"
    assert calls["n"] == 3


def test_generate_raises_after_max_retries(monkeypatch):
    def always_fail(*a, **k):
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "post", always_fail)
    client = OllamaClient(max_retries=2, backoff_s=0)
    with pytest.raises(OllamaError):
        client.generate("m", "p")


def test_request_payload_shape(monkeypatch):
    captured = {}

    def capture(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", capture)
    client = OllamaClient(base_url="http://localhost:11434", timeout_s=99,
                          temperature=0.0, num_predict=256, backoff_s=0)
    client.generate("mymodel", "myprompt")
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["timeout"] == 99
    assert captured["json"]["model"] == "mymodel"
    assert captured["json"]["prompt"] == "myprompt"
    assert captured["json"]["stream"] is False
    assert captured["json"]["options"] == {"temperature": 0.0, "num_predict": 256}
