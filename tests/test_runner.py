"""Grid construction and resume-logic tests (no real Ollama)."""
import json

from typo_study.runner import build_trials, run_trials
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


def test_run_trials_writes_records_and_grades(tmp_path):
    results = tmp_path / "trials.jsonl"
    client = StubClient()
    run_trials(CONFIG, {"fake": FakeTask()}, client, results)
    records = [json.loads(l) for l in results.read_text().splitlines()]
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
    records = [json.loads(l) for l in results.read_text().splitlines()]
    assert all(not r["correct"] for r in records)
    assert all(r["empty"] for r in records)
