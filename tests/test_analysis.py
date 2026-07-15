"""Stats-function tests on synthetic data."""
import numpy as np
import pandas as pd

from typo_study.analysis import accuracy_table, fit_logit, wilson_ci


def _fake_df():
    rows = []
    rng = np.random.default_rng(0)
    for model, base in [("small", 0.6), ("big", 0.9)]:
        for severity in [0.0, 0.2]:
            for i in range(50):
                p = base - severity  # accuracy drops with severity
                rows.append({"phase": "main", "model": model, "task": "t",
                             "item_id": f"item_{i}",  # items reused across cells
                             "severity": severity,
                             "correct": bool(rng.random() < p),
                             "empty": bool(rng.random() < 0.1)})
    return pd.DataFrame(rows)


def test_wilson_ci_brackets_the_mean():
    lo, hi = wilson_ci(80, 100)
    assert lo < 0.8 < hi and 0 <= lo <= hi <= 1


def test_wilson_ci_nondegenerate_at_ceiling():
    lo, hi = wilson_ci(100, 100)
    assert hi == 1.0 and 0.9 < lo < 1.0    # non-zero width at 100% accuracy


def test_accuracy_table_shape_and_ordering():
    table = accuracy_table(_fake_df())
    assert set(table.columns) >= {"model", "severity", "accuracy", "ci_lo", "ci_hi", "n"}
    assert len(table) == 4
    big0 = table[(table.model == "big") & (table.severity == 0.0)].accuracy.iloc[0]
    big2 = table[(table.model == "big") & (table.severity == 0.2)].accuracy.iloc[0]
    assert big0 > big2


def test_accuracy_table_reports_empty_rate():
    df = _fake_df()
    table = accuracy_table(df)
    assert "empty_rate" in table.columns
    cell = df[(df.model == "big") & (df.severity == 0.0)]
    expected = cell["empty"].mean()
    got = table[(table.model == "big") & (table.severity == 0.0)].empty_rate.iloc[0]
    assert got == expected


def test_fit_logit_uses_clustered_errors():
    out = fit_logit(_fake_df())
    assert "cluster" in out


def test_fit_logit_survives_perfect_separation():
    rows = []
    for model, correct in [("good", True), ("bad", False)]:
        for severity in [0.0, 0.1, 0.2]:
            for i in range(20):
                rows.append({"phase": "main", "model": model, "task": "t",
                             "item_id": f"item_{i}", "severity": severity,
                             "correct": correct, "empty": False})
    out = fit_logit(pd.DataFrame(rows))  # must not raise
    assert "WARNING" in out
