"""Generate 50 deterministic instruction-following items (outputs are committed)."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

TOPICS = ["the ocean", "coffee", "space travel", "ancient Rome", "jazz music",
          "honey bees", "volcanoes", "the internet", "chess", "photosynthesis"]


def build_items() -> list[dict]:
    items = []
    for t_idx, topic in enumerate(TOPICS):
        n_words = 3 + (t_idx % 5)
        n_bullets = 2 + (t_idx % 4)
        specs = [
            ("word_count",
             f"Describe {topic} in exactly {n_words} words. Respond with only those words.",
             {"n": n_words}, ["exactly", "only", "words"]),
            ("json_keys",
             f"Respond with only a valid JSON object with keys \"name\" and \"year\", "
             f"describing a famous person related to {topic}.",
             {"keys": ["name", "year"]}, ["JSON", "only", "keys", "name", "year", "valid", "object"]),
            ("lowercase",
             f"Write one sentence about {topic} using only lowercase letters.",
             {}, ["lowercase", "letters", "only"]),
            ("starts_with",
             f"Write a sentence about {topic} that starts with the word \"Surprisingly\".",
             {"word": "Surprisingly"}, ["starts", "word", "Surprisingly"]),
            ("bullet_count",
             f"List exactly {n_bullets} facts about {topic} as a bulleted list, "
             f"each line starting with '- '.",
             {"n": n_bullets}, ["exactly", "bulleted", "list", "line", "starting"]),
        ]
        for c_idx, (ctype, prompt, params, protected) in enumerate(specs):
            items.append({
                "id": f"instr-{t_idx:02d}{c_idx}",
                "prompt": prompt,
                "check": {"type": ctype, **params},
                "protected": protected,
            })
    return items


if __name__ == "__main__":
    items = build_items()
    (DATA / "instructions.jsonl").write_text(
        "".join(json.dumps(it) + "\n" for it in items))
    print(f"wrote {len(items)} items")
