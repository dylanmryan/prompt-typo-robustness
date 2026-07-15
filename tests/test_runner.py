"""Grid construction and resume-logic tests (no real Ollama)."""
import json
from pathlib import Path

import pytest

from typo_study.ollama_client import OllamaError
from typo_study.runner import build_trials, run_trials, trial_key
from typo_study.typos import TYPO_TYPES

CONFIG = {
    "seed": 1,
    "severities": [0.0, 0.2],
    "models": ["m1"],
    "item_counts": {"fake": 4},
    "model_item_fraction": {},
    "typo_breakdown": {"severity": 0.2, "item_fraction": 0.5},
}


class FakeTask:
    name = "fake"

    def __init__(self):
        self.items = [{"id": f"f{i}"} for i in range(4)]
        self.items_by_id = {it["id"]: it for it in self.items}

    def build_prompt(self, item):
        return f"please answer question {item['id']} with the word yes"

    def grade(self, item, response):
        return response == "yes"

    def protected_tokens(self, item):
        return set()


class StubClient:
    def __init__(self, reply="yes"):
        self.calls = []
        self.reply = reply

    def generate(self, model, prompt):
        self.calls.append((model, prompt))
        return self.reply


def test_build_trials_counts():
    trials = build_trials(CONFIG, {"fake": FakeTask()})
    main = [t for t in trials if t["phase"] == "main"]
    breakdown = [t for t in trials if t["phase"] == "breakdown"]
    assert len(main) == 1 * 4 * 2                      # models * items * severities
    assert len(breakdown) == 1 * 2 * len(TYPO_TYPES)   # models * sampled items * types


def test_build_trials_respects_item_fraction():
    cfg = dict(CONFIG, model_item_fraction={"m1": 0.5})
    trials = build_trials(cfg, {"fake": FakeTask()})
    main = [t for t in trials if t["phase"] == "main"]
    assert len(main) == 1 * 2 * 2


def test_build_trials_deterministic():
    a = build_trials(CONFIG, {"fake": FakeTask()})
    b = build_trials(CONFIG, {"fake": FakeTask()})
    assert a == b


def test_breakdown_items_identical_across_models():
    cfg = dict(CONFIG, models=["m1", "m2"], model_item_fraction={"m2": 0.5})
    trials = build_trials(cfg, {"fake": FakeTask()})
    bd = [t for t in trials if t["phase"] == "breakdown"]
    per_model = {m: {t["item_id"] for t in bd if t["model"] == m} for m in ("m1", "m2")}
    assert per_model["m1"] == per_model["m2"] and per_model["m1"]


def test_trial_keys_unique():
    trials = build_trials(CONFIG, {"fake": FakeTask()})
    assert len({trial_key(t) for t in trials}) == len(trials)


def test_run_trials_writes_records_and_grades(tmp_path):
    results = tmp_path / "trials.jsonl"
    client = StubClient()
    run_trials(CONFIG, {"fake": FakeTask()}, client, results)
    records = [json.loads(line) for line in results.read_text().splitlines()]
    assert len(records) == 16 and all(r["correct"] for r in records)
    zero = [r for r in records if r["severity"] == 0.0]
    assert all(r["n_edits"] == 0 for r in zero)
    corrupted = [r for r in records if r["severity"] > 0 and r["phase"] == "main"]
    assert all(r["n_edits"] >= 1 for r in corrupted)


def test_run_trials_resumes_without_repeating(tmp_path):
    results = tmp_path / "trials.jsonl"
    first = StubClient()
    run_trials(CONFIG, {"fake": FakeTask()}, first, results)
    second = StubClient()
    run_trials(CONFIG, {"fake": FakeTask()}, second, results)
    assert second.calls == []
    assert len(results.read_text().splitlines()) == 16


def test_incorrect_and_empty_responses_flagged(tmp_path):
    results = tmp_path / "trials.jsonl"
    run_trials(CONFIG, {"fake": FakeTask()}, StubClient(reply="  "), results)
    records = [json.loads(line) for line in results.read_text().splitlines()]
    assert all(not r["correct"] for r in records)
    assert all(r["empty"] for r in records)


def test_resume_tolerates_truncated_last_line(tmp_path):
    results = tmp_path / "trials.jsonl"
    run_trials(CONFIG, {"fake": FakeTask()}, StubClient(), results)
    text = results.read_text()
    results.write_text(text[: len(text) - 20])          # torn final line
    run_trials(CONFIG, {"fake": FakeTask()}, StubClient(), results)
    keys = [json.loads(line)["key"] for line in results.read_text().splitlines()
            if line.strip() and line.endswith("}")]
    assert len(keys) == len(set(keys)) == 16


def test_aborts_after_consecutive_model_failures(tmp_path):
    class DeadClient:
        def __init__(self):
            self.calls = 0

        def generate(self, model, prompt):
            self.calls += 1
            raise OllamaError("model not found")

    client = DeadClient()
    with pytest.raises(RuntimeError):
        run_trials(CONFIG, {"fake": FakeTask()}, client, tmp_path / "t.jsonl")
    assert client.calls == 5


def test_ollama_error_skips_and_retries_on_resume(tmp_path):
    class FlakyClient:
        def generate(self, model, prompt):
            if "f2" in prompt:
                raise OllamaError("boom")
            return "yes"

    results = tmp_path / "trials.jsonl"
    run_trials(CONFIG, {"fake": FakeTask()}, FlakyClient(), results)
    # f2 has 2 main trials plus 4 breakdown trials (the seeded breakdown sample
    # is {f2, f3}); its 4 consecutive breakdown failures stay below the abort
    # threshold, so the run completes with the other 10 trials recorded.
    assert len(results.read_text().splitlines()) == 10
    run_trials(CONFIG, {"fake": FakeTask()}, StubClient(), results)
    assert len(results.read_text().splitlines()) == 16


def test_write_failures_abort_the_run(tmp_path, monkeypatch):
    results = tmp_path / "trials.jsonl"
    real_open = Path.open

    class ExplodingFile:
        def __init__(self, fh):
            self._fh = fh

        def write(self, *a):
            raise OSError("disk full")

        def flush(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self._fh.close()
            return False

    monkeypatch.setattr(
        Path, "open", lambda self, *a, **k: ExplodingFile(real_open(self, *a, **k)))
    with pytest.raises(OSError):
        run_trials(CONFIG, {"fake": FakeTask()}, StubClient(), results)
