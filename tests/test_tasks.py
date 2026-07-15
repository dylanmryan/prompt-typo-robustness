"""Grader and loader tests for the three evaluation tasks."""
import json
from pathlib import Path

import pytest

from typo_study.tasks import InstructionTask, MathTask, SentimentTask, get_task


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    (tmp_path / "gsm8k.jsonl").write_text("".join(json.dumps(it) + "\n" for it in [
        {"id": "gsm8k-000", "question": "Tom has 3 boxes of 4 pens. How many pens?",
         "answer": "12"},
        {"id": "gsm8k-001", "question": "A widget costs 1200 dollars. Total?",
         "answer": "1200"},
    ]))
    (tmp_path / "sst2.jsonl").write_text(json.dumps(
        {"id": "sst2-000", "sentence": "a delightful and moving film", "label": "positive"}) + "\n")
    (tmp_path / "instructions.jsonl").write_text("".join(json.dumps(it) + "\n" for it in [
        {"id": "instr-000", "prompt": "Describe coffee in exactly 3 words.",
         "check": {"type": "word_count", "n": 3}, "protected": ["exactly", "words"]},
        {"id": "instr-001", "prompt": "Respond with only a valid JSON object with keys \"name\" and \"year\".",
         "check": {"type": "json_keys", "keys": ["name", "year"]}, "protected": ["name", "year"]},
        {"id": "instr-002", "prompt": "Write one sentence using only lowercase letters.",
         "check": {"type": "lowercase"}, "protected": ["lowercase"]},
        {"id": "instr-003", "prompt": "Write a sentence that starts with the word \"Surprisingly\".",
         "check": {"type": "starts_with", "word": "Surprisingly"}, "protected": ["Surprisingly"]},
        {"id": "instr-004", "prompt": "List exactly 2 facts as a bulleted list.",
         "check": {"type": "bullet_count", "n": 2}, "protected": ["exactly"]},
    ]))
    return tmp_path


def test_math_load_and_prompt(data_dir):
    task = MathTask(data_dir)
    assert task.items[0]["id"] == "gsm8k-000"
    assert "How many pens?" in task.build_prompt(task.items[0])


@pytest.mark.parametrize("response,expected", [
    ("The answer is\n#### 12", True),
    ("#### 12.0", True),
    ("He has 3 boxes so 3 * 4 = 12", True),        # falls back to last number
    ("#### 13", False),
    ("I cannot solve this", False),
    ("", False),
    ("#### 1,200", False),
    ("The answer is **12**. See step 2 above.", True),
    ("#### **12** final", True),
])
def test_math_grading(data_dir, response, expected):
    task = MathTask(data_dir)
    assert task.grade(task.items[0], response) is expected


def test_math_answer_phrase_beats_trailing_number(data_dir):
    task = MathTask(data_dir)
    item = task.items_by_id["gsm8k-000"]
    response = "The final answer is 12. Let me know if you have questions about problem 42."
    assert task.grade(item, response) is True


def test_math_comma_grouped_marker_answer(data_dir):
    task = MathTask(data_dir)
    item = task.items_by_id["gsm8k-001"]
    assert task.grade(item, "#### 1,200") is True


def test_math_protected_tokens_empty(data_dir):
    task = MathTask(data_dir)
    assert task.protected_tokens(task.items[0]) == set()


@pytest.mark.parametrize("response,expected", [
    ("positive", True),
    ("Positive.", True),
    ("The sentiment is positive", True),
    ("negative", False),
    ("It could be positive or negative", False),   # ambiguous -> wrong
    ("no idea", False),
])
def test_sentiment_grading(data_dir, response, expected):
    task = SentimentTask(data_dir)
    assert task.grade(task.items[0], response) is expected


def test_sentiment_protected_tokens(data_dir):
    task = SentimentTask(data_dir)
    assert {"positive", "negative"} <= task.protected_tokens(task.items[0])


@pytest.mark.parametrize("item_id,response,expected", [
    ("instr-000", "hot bitter drink", True),
    ("instr-000", "a hot bitter drink", False),
    ("instr-001", '{"name": "Ada", "year": 1815}', True),
    ("instr-001", 'Sure! {"name": "Ada", "year": 1815}', True),
    ("instr-001", '{"name": "Ada"}', False),
    ("instr-001", "not json at all", False),
    ("instr-001", '{"name": "Ada", "year": 1815} Note: format is like {this}.', True),
    ("instr-002", "coffee is nice.", True),
    ("instr-002", "Coffee is nice.", False),
    ("instr-003", "Surprisingly, chess is old.", True),
    ("instr-003", "Chess is surprisingly old.", False),
    ("instr-003", '"Surprisingly, chess is old."', True),
    ("instr-003", "**Surprisingly**, chess is old.", True),
    ("instr-004", "- fact one\n- fact two", True),
    ("instr-004", "- only one fact", False),
])
def test_instruction_grading(data_dir, item_id, response, expected):
    task = InstructionTask(data_dir)
    item = task.items_by_id[item_id]
    assert task.grade(item, response) is expected


def test_instruction_protected_tokens(data_dir):
    task = InstructionTask(data_dir)
    item = task.items_by_id["instr-001"]
    assert task.protected_tokens(item) == set(item["protected"])


def test_get_task_registry(data_dir):
    assert isinstance(get_task("math", data_dir), MathTask)
    assert isinstance(get_task("sentiment", data_dir), SentimentTask)
    assert isinstance(get_task("instructions", data_dir), InstructionTask)
