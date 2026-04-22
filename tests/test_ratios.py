"""
Test Suite 1: Financial Ratio Calculations
Validates that ratio engineering produces correct, bounded values.
Run: python tests/test_ratios.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_ratios():
    path = os.path.join(DATA_PATH, "fdic_financials_with_ratios.csv")
    assert os.path.exists(path), f"Ratio file not found at {path}"
    return pd.read_csv(path)


def test_file_exists_and_nonempty():
    df = load_ratios()
    assert len(df) > 0, "Ratio file is empty"
    print(f"  PASS: {len(df)} rows loaded")


def test_required_columns_present():
    df = load_ratios()
    required = ["bank_name", "liquidity_ratio", "debt_to_equity",
                 "interest_coverage", "loan_to_deposit"]
    missing = [c for c in required if c not in df.columns]
    assert len(missing) == 0, f"Missing columns: {missing}"
    print(f"  PASS: All {len(required)} required columns present")


def test_no_infinite_values():
    df = load_ratios()
    ratio_cols = ["liquidity_ratio", "debt_to_equity", "interest_coverage", "loan_to_deposit"]
    for col in ratio_cols:
        if col in df.columns:
            inf_count = np.isinf(df[col].dropna()).sum()
            assert inf_count == 0, f"Column {col} has {inf_count} infinite values"
    print("  PASS: No infinite values in any ratio column")


def test_liquidity_ratio_bounded():
    """Liquidity ratio = (Cash + Securities) / Deposits — should be >= 0."""
    df = load_ratios()
    if "liquidity_ratio" in df.columns:
        valid = df["liquidity_ratio"].dropna()
        assert (valid >= 0).all(), "Liquidity ratio has negative values"
        print(f"  PASS: Liquidity ratio range [{valid.min():.4f}, {valid.max():.4f}]")


def test_debt_to_equity_positive():
    """Debt-to-equity = Total Liabilities / Total Equity — should be > 0 for operating banks."""
    df = load_ratios()
    if "debt_to_equity" in df.columns:
        valid = df["debt_to_equity"].dropna()
        assert (valid > 0).all(), "Debt-to-equity has non-positive values"
        print(f"  PASS: Debt-to-equity range [{valid.min():.2f}, {valid.max():.2f}]")


def test_bank_count():
    """We should have 19 target banks."""
    df = load_ratios()
    n_banks = df["bank_name"].nunique()
    assert n_banks >= 15, f"Only {n_banks} banks found, expected ~19"
    print(f"  PASS: {n_banks} unique banks")


def test_year_coverage():
    """Data should span 2018-2023."""
    df = load_ratios()
    if "year" in df.columns:
        years = sorted(df["year"].dropna().unique())
        assert min(years) <= 2018, f"Earliest year is {min(years)}, expected <= 2018"
        assert max(years) >= 2023, f"Latest year is {max(years)}, expected >= 2023"
        print(f"  PASS: Year range {int(min(years))}–{int(max(years))}")


if __name__ == "__main__":
    tests = [
        test_file_exists_and_nonempty,
        test_required_columns_present,
        test_no_infinite_values,
        test_liquidity_ratio_bounded,
        test_debt_to_equity_positive,
        test_bank_count,
        test_year_coverage,
    ]

    print("=" * 60)
    print("TEST SUITE: Financial Ratio Calculations")
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
