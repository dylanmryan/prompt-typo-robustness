"""Download and freeze small evaluation subsets (run once; outputs are committed)."""
import json
import re
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
GSM8K_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
             "master/grade_school_math/data/test.jsonl")
SST2_URL = ("https://datasets-server.huggingface.co/rows?dataset=stanfordnlp%2Fsst2"
            "&config=default&split=validation&offset=0&length=100")


def fetch_gsm8k(n: int = 75) -> None:
    raw = urllib.request.urlopen(GSM8K_URL, timeout=60).read().decode()
    items = []
    for i, line in enumerate(raw.strip().splitlines()[:n]):
        row = json.loads(line)
        gold = re.search(r"####\s*(.+)", row["answer"]).group(1)
        items.append({
            "id": f"gsm8k-{i:03d}",
            "question": " ".join(row["question"].split()),
            "answer": gold.strip().replace(",", ""),
        })
    _write(DATA / "gsm8k.jsonl", items)


def fetch_sst2() -> None:
    raw = json.loads(urllib.request.urlopen(SST2_URL, timeout=60).read().decode())
    items = []
    for i, row in enumerate(raw["rows"]):
        items.append({
            "id": f"sst2-{i:03d}",
            "sentence": " ".join(row["row"]["sentence"].split()),
            "label": "positive" if row["row"]["label"] == 1 else "negative",
        })
    _write(DATA / "sst2.jsonl", items)


def _write(path: Path, items: list[dict]) -> None:
    path.write_text("".join(json.dumps(it) + "\n" for it in items))
    print(f"wrote {len(items)} items to {path}")


if __name__ == "__main__":
    fetch_gsm8k()
    fetch_sst2()
