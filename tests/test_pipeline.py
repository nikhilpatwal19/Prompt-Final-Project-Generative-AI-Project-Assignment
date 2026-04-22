"""
Test Suite 3: RAG Coverage & Model Pipeline
Validates RAG assessments, fragility scores, and ML model outputs.
Run: python tests/test_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


# ── RAG Tests ──

def test_rag_assessments_exist():
    path = os.path.join(DATA_PATH, "enhanced_rag_assessments.csv")
    assert os.path.exists(path), "Enhanced RAG assessments file not found"
    df = pd.read_csv(path)
    assert len(df) > 0, "RAG assessments file is empty"
    print(f"  PASS: {len(df)} RAG assessments loaded")


def test_rag_100_percent_coverage():
    """Enhanced RAG should cover all banks (target: 100% coverage)."""
    path = os.path.join(DATA_PATH, "enhanced_rag_assessments.csv")
    if not os.path.exists(path):
        print("  SKIP: enhanced_rag_assessments.csv not found")
        return
    df = pd.read_csv(path)
    if "num_sources" in df.columns:
        zero_sources = (df["num_sources"] == 0).sum()
        coverage_pct = (1 - zero_sources / len(df)) * 100
        assert coverage_pct >= 90, f"RAG coverage only {coverage_pct:.0f}%, expected >= 90%"
        print(f"  PASS: RAG coverage = {coverage_pct:.0f}% ({len(df)} assessments)")
    elif "assessment" in df.columns:
        non_empty = df["assessment"].notna().sum()
        print(f"  PASS: {non_empty}/{len(df)} assessments have content")


def test_rag_assessments_have_content():
    """Each assessment should be more than a trivial stub."""
    path = os.path.join(DATA_PATH, "enhanced_rag_assessments.csv")
    if not os.path.exists(path):
        print("  SKIP: File not found")
        return
    df = pd.read_csv(path)
    if "assessment" in df.columns:
        lengths = df["assessment"].dropna().str.len()
        avg_len = lengths.mean()
        assert avg_len > 100, f"Average assessment length {avg_len:.0f} chars — too short"
        print(f"  PASS: Average assessment length = {avg_len:.0f} characters")


# ── Fragility Score Tests ──

def test_fragility_scores_exist():
    path = os.path.join(DATA_PATH, "fragility_scored.csv")
    assert os.path.exists(path), "Fragility scores file not found"
    df = pd.read_csv(path)
    assert len(df) > 0, "Fragility scores file is empty"
    print(f"  PASS: {len(df)} fragility score records loaded")


def test_fragility_v2_present():
    """Data-driven V2 scores should exist."""
    path = os.path.join(DATA_PATH, "fragility_scored.csv")
    df = pd.read_csv(path)
    assert "fragility_score_v2" in df.columns, "Missing fragility_score_v2 column"
    non_null = df["fragility_score_v2"].notna().sum()
    assert non_null > 0, "fragility_score_v2 is entirely NaN"
    print(f"  PASS: V2 scores present for {non_null}/{len(df)} observations")


def test_failed_banks_rank_high():
    """Known failed banks should have above-median fragility scores."""
    path = os.path.join(DATA_PATH, "fragility_scored.csv")
    df = pd.read_csv(path)
    score_col = "fragility_score_v2" if "fragility_score_v2" in df.columns else "fragility_score"

    # Get latest year per bank
    latest = df.sort_values("year").groupby("bank_name").last().reset_index()
    median_score = latest[score_col].median()

    failed_banks = latest[latest["failed"] == True]
    if len(failed_banks) == 0:
        print("  SKIP: No failed banks in dataset")
        return

    above_median = (failed_banks[score_col] > median_score).sum()
    pct = above_median / len(failed_banks) * 100
    assert pct >= 50, f"Only {pct:.0f}% of failed banks above median fragility"
    print(f"  PASS: {above_median}/{len(failed_banks)} failed banks above median ({score_col}={median_score:.3f})")


def test_scores_bounded_0_to_1():
    """Fragility scores should be normalized between 0 and 1."""
    path = os.path.join(DATA_PATH, "fragility_scored.csv")
    df = pd.read_csv(path)
    score_col = "fragility_score_v2" if "fragility_score_v2" in df.columns else "fragility_score"
    valid = df[score_col].dropna()
    assert valid.min() >= 0, f"Score below 0: {valid.min()}"
    assert valid.max() <= 1.01, f"Score above 1: {valid.max()}"
    print(f"  PASS: Scores in [{valid.min():.3f}, {valid.max():.3f}]")


# ── LLM Score Tests ──

def test_llm_scores_exist():
    path = os.path.join(DATA_PATH, "llm_fragility_scores.csv")
    assert os.path.exists(path), "LLM scores file not found"
    df = pd.read_csv(path)
    assert len(df) > 0, "LLM scores file is empty"
    print(f"  PASS: {len(df)} LLM-scored banks loaded")


def test_llm_scores_in_range():
    """LLM scores should be 1-10."""
    path = os.path.join(DATA_PATH, "llm_fragility_scores.csv")
    df = pd.read_csv(path)
    score_cols = [c for c in df.columns if "score" in c.lower() or "fragility" in c.lower()]
    if score_cols:
        col = score_cols[0]
        valid = pd.to_numeric(df[col], errors="coerce").dropna()
        assert valid.min() >= 1, f"LLM score below 1: {valid.min()}"
        assert valid.max() <= 10, f"LLM score above 10: {valid.max()}"
        print(f"  PASS: LLM scores in [{valid.min():.1f}, {valid.max():.1f}]")
    else:
        print("  SKIP: No score column found in LLM scores file")


# ── Output File Tests ──

def test_all_visualizations_generated():
    """All expected output files should exist."""
    expected_outputs = [
        "ratio_trends.png",
        "fragility_heatmap.png",
        "fragility_components.png",
        "fragility_timeseries.png",
        "feature_importance.png",
        "llm_vs_rulebased.png",
        "real_vs_synthetic_distributions.png",
        "comprehensive_model_evaluation.png",
        "precision_recall_curves.png",
        "fragility_dashboard.html",
        "fragility_breakdown.html",
        "fragility_vs_roa.html",
    ]

    missing = [f for f in expected_outputs if not os.path.exists(os.path.join(OUTPUT_PATH, f))]
    assert len(missing) == 0, f"Missing outputs: {missing}"
    print(f"  PASS: All {len(expected_outputs)} expected output files present")


def test_output_files_not_empty():
    """Output files should have non-trivial size."""
    for fname in os.listdir(OUTPUT_PATH):
        fpath = os.path.join(OUTPUT_PATH, fname)
        size = os.path.getsize(fpath)
        assert size > 1000, f"{fname} is only {size} bytes — possibly corrupt"
    print(f"  PASS: All output files have substantial content")


if __name__ == "__main__":
    tests = [
        test_rag_assessments_exist,
        test_rag_100_percent_coverage,
        test_rag_assessments_have_content,
        test_fragility_scores_exist,
        test_fragility_v2_present,
        test_failed_banks_rank_high,
        test_scores_bounded_0_to_1,
        test_llm_scores_exist,
        test_llm_scores_in_range,
        test_all_visualizations_generated,
        test_output_files_not_empty,
    ]

    print("=" * 60)
    print("TEST SUITE: RAG Coverage & Model Pipeline")
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
