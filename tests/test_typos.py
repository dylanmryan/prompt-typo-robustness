"""Tests for the seeded QWERTY typo engine."""
from typo_study.typos import TYPO_TYPES, corrupt

TEXT = ("Sally bought seventeen apples from the market and gave four "
        "apples to her brother before walking home yesterday evening")  # 19 words, 18 eligible ("to" too short)


def test_deterministic_given_same_seed():
    a = corrupt(TEXT, 0.2, seed=7)
    b = corrupt(TEXT, 0.2, seed=7)
    assert a.text == b.text and len(a.edits) == len(b.edits)


def test_different_seeds_differ():
    assert corrupt(TEXT, 0.2, seed=1).text != corrupt(TEXT, 0.2, seed=2).text


def test_severity_zero_is_identity():
    r = corrupt(TEXT, 0.0, seed=7)
    assert r.text == TEXT and r.edits == []


def test_severity_fraction_of_eligible_words():
    eligible = [w for w in TEXT.split() if w.isalpha() and len(w) >= 3]
    r = corrupt(TEXT, 0.25, seed=7)
    assert len(r.edits) == round(len(eligible) * 0.25)


def test_minimum_one_edit_for_nonzero_severity():
    r = corrupt("please solve this", 0.01, seed=7)
    assert len(r.edits) == 1


def test_protected_tokens_never_modified():
    for seed in range(20):
        r = corrupt(TEXT, 1.0, seed=seed, protected={"apples", "market"})
        assert "apples" in r.text and "market" in r.text
        assert all(e.original.lower() not in {"apples", "market"} for e in r.edits)


def test_digits_never_modified():
    text = "add 1234 and 567 then report the total sum please"
    for seed in range(20):
        r = corrupt(text, 1.0, seed=seed)
        assert "1234" in r.text and "567" in r.text


def test_punctuation_attached_words_are_eligible_and_punct_preserved():
    r = corrupt("Firstly, remove the wrapping carefully.", 1.0, seed=3)
    assert r.text.endswith(".") and r.text.split()[0].endswith(",")
    assert len(r.edits) >= 3


def test_edits_record_matches_text():
    r = corrupt(TEXT, 0.3, seed=11)
    for e in r.edits:
        assert e.corrupted in r.text
        assert e.typo_type in TYPO_TYPES


def test_single_typo_type_restriction():
    r = corrupt(TEXT, 0.5, seed=5, typo_types=("deletion",))
    assert all(e.typo_type == "deletion" for e in r.edits)
    assert all(len(e.corrupted) == len(e.original) - 1 for e in r.edits)


def test_invalid_severity_raises():
    import pytest
    with pytest.raises(ValueError):
        corrupt(TEXT, 1.5, seed=0)
