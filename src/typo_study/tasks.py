"""Task loaders, prompt templates, graders, and protected-token lists."""
from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _to_number(s: str) -> float:
    return float(s.replace(",", "").strip().rstrip("."))


class _BaseTask:
    filename = ""

    def __init__(self, data_dir: Path | None = None):
        data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.items: list[dict] = _load_jsonl(data_dir / self.filename)
        self.items_by_id: dict[str, dict] = {it["id"]: it for it in self.items}


_MARKER_ANSWER_RE = re.compile(r"####\s*\**\s*(" + _NUM_RE.pattern + ")")
_PHRASE_ANSWER_RE = re.compile(
    r"answer(?:\s+is)?\s*:?\s*\**\s*(" + _NUM_RE.pattern + ")", re.IGNORECASE)


def _extract_math_answer(response: str) -> str | None:
    """Extract the candidate final answer from a model response.

    Cascade: the explicit '#### <number>' marker wins; else the last
    'answer is <number>' phrase; else the last number anywhere (so stray
    numbers after the stated answer cannot override it).
    """
    m = _MARKER_ANSWER_RE.search(response)
    if m:
        return m.group(1)
    phrase_matches = _PHRASE_ANSWER_RE.findall(response)
    if phrase_matches:
        return phrase_matches[-1]
    numbers = _NUM_RE.findall(response)
    return numbers[-1] if numbers else None


class MathTask(_BaseTask):
    name = "math"
    filename = "gsm8k.jsonl"

    def build_prompt(self, item: dict) -> str:
        return ("Solve the following math problem. Show your reasoning, then end "
                "your response with the final numeric answer on its own line in the "
                "form '#### <number>'.\n\n" + item["question"])

    def grade(self, item: dict, response: str) -> bool:
        cand = _extract_math_answer(response)
        if cand is None:
            return False
        try:
            return _to_number(cand) == _to_number(item["answer"])
        except ValueError:
            return False

    def protected_tokens(self, item: dict) -> set[str]:
        return set()  # digits are protected by the typo engine itself


class SentimentTask(_BaseTask):
    name = "sentiment"
    filename = "sst2.jsonl"

    def build_prompt(self, item: dict) -> str:
        return ("Classify the sentiment of the following movie review as positive "
                "or negative. Reply with exactly one word: positive or negative.\n\n"
                "Review: " + item["sentence"])

    def grade(self, item: dict, response: str) -> bool:
        # Intentionally strict: the prompt demands exactly one label word, so a
        # response mentioning both labels is graded as an instruction failure.
        found = set(re.findall(r"\b(positive|negative)\b", response.lower()))
        return found == {item["label"]}

    def protected_tokens(self, item: dict) -> set[str]:
        return {"positive", "negative"}


class InstructionTask(_BaseTask):
    name = "instructions"
    filename = "instructions.jsonl"

    def build_prompt(self, item: dict) -> str:
        return item["prompt"]

    def grade(self, item: dict, response: str) -> bool:
        check = item["check"]
        checker = _CHECKERS[check["type"]]
        return bool(checker(response, check))

    def protected_tokens(self, item: dict) -> set[str]:
        return set(item["protected"])


def _check_json_keys(response: str, check: dict) -> bool:
    decoder = json.JSONDecoder()
    for start in (m.start() for m in re.finditer(r"\{", response)):
        try:
            obj, _ = decoder.raw_decode(response[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return set(obj.keys()) == set(check["keys"])
    return False


_CHECKERS = {
    "word_count": lambda r, c: len(r.split()) == c["n"],
    "json_keys": _check_json_keys,
    "lowercase": lambda r, c: r == r.lower() and any(ch.isalpha() for ch in r),
    "starts_with": lambda r, c: r.strip().lstrip("\"'“‘*").lower().startswith(c["word"].lower()),
    "bullet_count": lambda r, c: sum(
        1 for ln in r.splitlines() if ln.strip().startswith("- ")) == c["n"],
}

_REGISTRY = {"math": MathTask, "sentiment": SentimentTask, "instructions": InstructionTask}


def get_task(name: str, data_dir: Path | None = None):
    return _REGISTRY[name](data_dir)
