"""Seeded, QWERTY-realistic typo generation. Pure functions, no I/O."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

QWERTY_NEIGHBORS = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh",
    "u": "yij", "i": "uok", "o": "ipl", "p": "ol",
    "a": "qsz", "s": "awdx", "d": "sefc", "f": "drgv", "g": "fthb",
    "h": "gyjn", "j": "hukm", "k": "jil", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}

TYPO_TYPES = ("substitution", "transposition", "deletion", "doubling")

_TOKEN_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)


@dataclass
class Edit:
    token_index: int  # word ordinal: index into text.split()
    original: str
    corrupted: str
    typo_type: str


@dataclass
class CorruptionResult:
    text: str
    edits: list[Edit] = field(default_factory=list)


def _eligible(core: str, protected: set[str]) -> bool:
    return core.isalpha() and len(core) >= 3 and core.lower() not in protected


def _apply_typo(word: str, typo_type: str, rng: random.Random) -> str:
    if typo_type == "substitution":
        positions = [i for i, ch in enumerate(word) if ch.lower() in QWERTY_NEIGHBORS]
        if not positions:  # no QWERTY-mapped letters (e.g. non-ASCII word)
            return _apply_typo(word, "doubling", rng)
        i = rng.choice(positions)
        repl = rng.choice(QWERTY_NEIGHBORS[word[i].lower()])
        if word[i].isupper():
            repl = repl.upper()
        return word[:i] + repl + word[i + 1:]
    if typo_type == "transposition":
        positions = [i for i in range(len(word) - 1) if word[i] != word[i + 1]]
        if not positions:  # uniform word like "aaa" cannot change under transposition
            return _apply_typo(word, "doubling", rng)
        i = rng.choice(positions)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if typo_type == "deletion":
        i = rng.randrange(len(word))
        return word[:i] + word[i + 1:]
    if typo_type == "doubling":
        i = rng.randrange(len(word))
        return word[: i + 1] + word[i] + word[i + 1:]
    raise ValueError(f"unknown typo type: {typo_type}")


def corrupt(
    text: str,
    severity: float,
    seed: int,
    protected: frozenset[str] | set[str] = frozenset(),
    typo_types: tuple[str, ...] = TYPO_TYPES,
) -> CorruptionResult:
    """Corrupt `severity` fraction of eligible words in `text`, deterministically.

    Eligibility: only purely alphabetic words of length >= 3 are corrupted
    (punctuation attached to a word is stripped before the check and preserved
    in the output). Contractions ("don't") and hyphenated words ("well-known")
    are intentionally left untouched, and digits are never corrupted, so
    corruptions degrade surface form without changing ground-truth answers.

    For any severity > 0 at least one eligible word is corrupted, so nominal
    severity is inflated toward 1/len(eligible) for short texts (floor effect).

    Args:
        text: the prompt to corrupt.
        severity: fraction of eligible words to corrupt, in [0, 1].
        seed: RNG seed; identical inputs and seed give identical output.
        protected: words (case-insensitive) that must never be corrupted.
        typo_types: subset of TYPO_TYPES to draw from; must be non-empty.

    Raises:
        ValueError: if severity is outside [0, 1] or typo_types is empty.
    """
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be in [0, 1]")
    if not typo_types:
        raise ValueError("typo_types must be non-empty")
    rng = random.Random(seed)
    tokens = re.split(r"(\s+)", text)
    protected_lower = {p.lower() for p in protected}
    candidates = []
    for idx in range(0, len(tokens), 2):  # even indices are non-whitespace tokens
        core = _TOKEN_RE.match(tokens[idx]).group(2)
        if _eligible(core, protected_lower):
            candidates.append(idx)
    if severity == 0.0 or not candidates:
        return CorruptionResult(text=text)
    n = min(len(candidates), max(1, round(len(candidates) * severity)))
    chosen = sorted(rng.sample(candidates, n))
    edits = []
    for idx in chosen:
        prefix, core, suffix = _TOKEN_RE.match(tokens[idx]).groups()
        typo_type = rng.choice(typo_types)
        corrupted = _apply_typo(core, typo_type, rng)
        tokens[idx] = prefix + corrupted + suffix
        edits.append(Edit(token_index=idx // 2, original=core, corrupted=corrupted, typo_type=typo_type))
    return CorruptionResult(text="".join(tokens), edits=edits)
