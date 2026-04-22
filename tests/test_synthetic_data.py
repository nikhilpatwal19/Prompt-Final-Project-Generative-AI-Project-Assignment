"""
Test Suite 2: Synthetic Data Validation
Ensures generated synthetic profiles are realistic, labeled, and properly structured.
Run: python tests/test_synthetic_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_synthetic():
    path = os.path.join(DATA_PATH, "synthetic_bank_profiles.csv")
    assert os.path.exists(path), f"Synthetic data file not found at {path}"
    return pd.read_csv(path)


def load_real():
    path = os.path.join(DATA_PATH, "fdic_financials_with_ratios.csv")
    assert os.path.exists(path), f"Real data file not found at {path}"
    return pd.read_csv(path)


def test_synthetic_file_exists():
    df = load_synthetic()
    assert len(df) > 0, "Synthetic file is empty"
    print(f"  PASS: {len(df)} synthetic profiles loaded")


def test_all_labeled_synthetic():
    """Every synthetic row must have is_synthetic=True — ethical requirement."""
    df = load_synthetic()
    assert "is_synthetic" in df.columns, "Missing 'is_synthetic' column"
    assert df["is_synthetic"].all(), "Some synthetic rows are NOT labeled as synthetic"
    print(f"  PASS: All {len(df)} rows correctly labeled is_synthetic=True")


def test_archetype_coverage():
    """Should have all 5 fragility archetypes: A, B, C, D, E."""
    df = load_synthetic()
    if "archetype" in df.columns:
        archetypes = set(df["archetype"].unique())
        expected = {"A", "B", "C", "D", "E"}
        missing = expected - archetypes
        assert len(missing) == 0, f"Missing archetypes: {missing}"
        print(f"  PASS: All 5 archetypes present — {sorted(archetypes)}")
    else:
        print("  SKIP: No 'archetype' column found")


def test_expected_count():
    """Should have ~80 synthetic profiles (50 distressed + 30 stable)."""
    df = load_synthetic()
    assert len(df) >= 50, f"Only {len(df)} profiles, expected ~80"
    print(f"  PASS: {len(df)} profiles generated (target: 80)")


def test_distress_label_distribution():
    """Should have both distressed and stable synthetic profiles."""
    df = load_synthetic()
    if "distressed" in df.columns:
        dist_counts = df["distressed"].value_counts()
        assert True in dist_counts.index or 1 in dist_counts.index, "No distressed profiles found"
        assert False in dist_counts.index or 0 in dist_counts.index, "No stable profiles found"
        n_dist = dist_counts.get(True, 0) + dist_counts.get(1, 0)
        n_stable = dist_counts.get(False, 0) + dist_counts.get(0, 0)
        print(f"  PASS: {n_dist} distressed, {n_stable} stable profiles")


def test_ratio_values_realistic():
    """Synthetic ratio values should be within plausible financial ranges."""
    df = load_synthetic()

    checks = {
        "liquidity_ratio": (0, 2.0),
        "debt_to_equity": (0, 50),
        "interest_coverage": (0, 500),
        "loan_to_deposit": (0, 3.0),
    }

    for col, (lo, hi) in checks.items():
        if col in df.columns:
            valid = df[col].dropna()
            out_of_range = ((valid < lo) | (valid > hi)).sum()
            pct = out_of_range / len(valid) * 100 if len(valid) > 0 else 0
            assert pct < 20, f"{col}: {pct:.1f}% of values outside [{lo}, {hi}]"

    print("  PASS: Synthetic ratios within plausible financial ranges")


def test_no_nan_in_key_columns():
    """Key ratio columns should not be all NaN."""
    df = load_synthetic()
    key_cols = ["liquidity_ratio", "debt_to_equity", "interest_coverage"]
    for col in key_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            assert non_null > 0, f"Column {col} is entirely NaN"
    print("  PASS: Key columns have non-null values")


def test_augmented_data_combines_correctly():
    """Augmented dataset should be larger than the merged feature matrix (real observations only)."""
    aug_path = os.path.join(DATA_PATH, "augmented_training_data.csv")
    merged_path = os.path.join(DATA_PATH, "merged_feature_matrix.csv")
    if os.path.exists(aug_path):
        aug = pd.read_csv(aug_path)
        # Compare against merged matrix (112 real obs), not raw FDIC (442 rows)
        if os.path.exists(merged_path):
            real = pd.read_csv(merged_path)
        else:
            real = load_synthetic()  # fallback: at least check aug > synthetic alone
        assert len(aug) > len(real), (
            f"Augmented ({len(aug)}) should be larger than real ({len(real)})"
        )
        print(f"  PASS: Augmented set ({len(aug)}) > real observations ({len(real)})")
    else:
        print("  SKIP: augmented_training_data.csv not found")


if __name__ == "__main__":
    tests = [
        test_synthetic_file_exists,
        test_all_labeled_synthetic,
        test_archetype_coverage,
        test_expected_count,
        test_distress_label_distribution,
        test_ratio_values_realistic,
        test_no_nan_in_key_columns,
        test_augmented_data_combines_correctly,
    ]

    print("=" * 60)
    print("TEST SUITE: Synthetic Data Validation")
    print("=" * 60)

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__} — {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test.__name__} — {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)
