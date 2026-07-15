"""Stats-function tests on synthetic data."""
import numpy as np
import pandas as pd

from typo_study.analysis import accuracy_table, bootstrap_ci


def _fake_df():
    rows = []
    rng = np.random.default_rng(0)
    for model, base in [("small", 0.6), ("big", 0.9)]:
        for severity in [0.0, 0.2]:
            for i in range(50):
                p = base - severity  # accuracy drops with severity
                rows.append({"phase": "main", "model": model, "task": "t",
                             "severity": severity,
                             "correct": bool(rng.random() < p)})
    return pd.DataFrame(rows)


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    values = [1] * 80 + [0] * 20
    lo, hi = bootstrap_ci(values, seed=0)
    assert lo < 0.8 < hi and 0 <= lo <= hi <= 1
    assert (lo, hi) == bootstrap_ci(values, seed=0)


def test_accuracy_table_shape_and_ordering():
    table = accuracy_table(_fake_df())
    assert set(table.columns) >= {"model", "severity", "accuracy", "ci_lo", "ci_hi", "n"}
    assert len(table) == 4
    big0 = table[(table.model == "big") & (table.severity == 0.0)].accuracy.iloc[0]
    big2 = table[(table.model == "big") & (table.severity == 0.2)].accuracy.iloc[0]
    assert big0 > big2
