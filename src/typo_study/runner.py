"""Config-driven, resumable experiment runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import yaml

from .ollama_client import OllamaClient, OllamaError
from .tasks import get_task
from .typos import TYPO_TYPES, corrupt

TASK_NAMES = ("math", "sentiment", "instructions")

# Abort if one model fails this many trials in a row (missing model, dead server).
MAX_CONSECUTIVE_FAILURES = 5


def trial_key(t: dict) -> str:
    return "|".join([t["phase"], t["model"], t["task"], t["item_id"],
                     f"{t['severity']:.2f}", t["typo_type"]])


def _typo_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode()).hexdigest()
    return int(digest[:12], 16)


def build_trials(config: dict, task_map: dict) -> list[dict]:
    trials = []
    fractions = config.get("model_item_fraction") or {}
    bd = config["typo_breakdown"]
    for task_name, task in task_map.items():
        items = task.items[: config["item_counts"][task_name]]
        rng = random.Random(f"{config['seed']}:{task_name}")
        # Prefix truncation is intentional and deterministic: every model's item
        # subset is a prefix of `items`, so sampling breakdown items from the
        # largest prefix common to all models guarantees every model runs the
        # exact same breakdown items, keeping typo-type results comparable.
        pool_len = min(max(1, round(len(items) * fractions.get(m, 1.0)))
                       for m in config["models"])
        pool = items[:pool_len]
        k = min(max(1, round(len(items) * bd["item_fraction"])), pool_len)
        breakdown_ids = {it["id"] for it in rng.sample(pool, k)}
        for model in config["models"]:
            frac = fractions.get(model, 1.0)
            use = items[: max(1, round(len(items) * frac))]
            for item in use:
                for severity in config["severities"]:
                    trials.append({"phase": "main", "model": model, "task": task_name,
                                   "item_id": item["id"], "severity": float(severity),
                                   "typo_type": "mixed"})
            for item in use:
                if item["id"] not in breakdown_ids:
                    continue
                for tt in TYPO_TYPES:
                    trials.append({"phase": "breakdown", "model": model, "task": task_name,
                                   "item_id": item["id"],
                                   "severity": float(bd["severity"]), "typo_type": tt})
    return trials


def _load_done(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    done = set()
    for lineno, line in enumerate(results_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["key"])
        except (json.JSONDecodeError, KeyError):
            print(f"WARNING: skipping malformed results line {lineno} (truncated write?)")
    return done


def _repair_torn_tail(results_path: Path) -> None:
    """A killed process can leave a torn final line; keep appends on fresh lines."""
    if not results_path.exists() or not results_path.stat().st_size:
        return
    with results_path.open("rb") as f:
        f.seek(-1, 2)
        if f.read(1) != b"\n":
            with results_path.open("a") as fa:
                fa.write("\n")


def run_trials(config: dict, task_map: dict, client, results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    _repair_torn_tail(results_path)
    done = _load_done(results_path)
    trials = build_trials(config, task_map)
    todo = [t for t in trials if trial_key(t) not in done]
    print(f"{len(trials)} trials total, {len(done)} done, {len(todo)} to run")
    consecutive_failures: dict[str, int] = {}
    with results_path.open("a") as f:
        for i, t in enumerate(todo):
            key = trial_key(t)
            task = task_map[t["task"]]
            item = task.items_by_id[t["item_id"]]
            prompt = task.build_prompt(item)
            types = TYPO_TYPES if t["typo_type"] == "mixed" else (t["typo_type"],)
            corrupted = corrupt(prompt, t["severity"], _typo_seed(config["seed"], key),
                                task.protected_tokens(item), types)
            start = time.time()
            try:
                response = client.generate(t["model"], corrupted.text)
            except OllamaError as err:
                n_failed = consecutive_failures.get(t["model"], 0) + 1
                consecutive_failures[t["model"]] = n_failed
                print(f"SKIP {key}: {err}")
                if n_failed >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(
                        f"model {t['model']} failed {MAX_CONSECUTIVE_FAILURES} "
                        "consecutive trials; aborting (is it pulled?)")
                continue
            consecutive_failures[t["model"]] = 0
            try:
                record = {**t, "key": key, "prompt": corrupted.text,
                          "n_edits": len(corrupted.edits), "response": response,
                          "correct": bool(task.grade(item, response)),
                          "empty": not response.strip(),
                          "latency_s": round(time.time() - start, 2), "ts": time.time()}
                f.write(json.dumps(record) + "\n")
                f.flush()
            except Exception as exc:
                # The response is lost, but the run survives; resume retries the trial.
                print(f"ERROR {key}: {exc!r}")
                continue
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(todo)} done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the typo robustness experiment")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    gen = config.get("generation", {})
    client = OllamaClient(timeout_s=gen.get("timeout_s", 180),
                          temperature=gen.get("temperature", 0.0),
                          num_predict=gen.get("num_predict", 512))
    task_map = {name: get_task(name) for name in TASK_NAMES}
    run_trials(config, task_map, client, Path(config["results_path"]))


if __name__ == "__main__":
    main()
