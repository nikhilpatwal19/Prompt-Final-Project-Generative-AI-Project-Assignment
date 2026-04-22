"""
Run all test suites for the AI Financial Fragility Detector.

Usage:
    python tests/run_all_tests.py

This script runs all three test suites:
  1. Financial Ratio Calculations
  2. Synthetic Data Validation
  3. RAG Coverage & Model Pipeline
"""

import subprocess
import sys
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

suites = [
    "test_ratios.py",
    "test_synthetic_data.py",
    "test_pipeline.py",
]

print("=" * 60)
print("AI FINANCIAL FRAGILITY DETECTOR — FULL TEST SUITE")
print("=" * 60)
print()

total_pass = 0
total_fail = 0

for suite in suites:
    path = os.path.join(TESTS_DIR, suite)
    print(f"Running {suite}...")
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Parse results line
    for line in result.stdout.split("\n"):
        if "RESULTS:" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "passed,":
                    total_pass += int(parts[i - 1])
                if p == "failed":
                    total_fail += int(parts[i - 1])

print("=" * 60)
print(f"GRAND TOTAL: {total_pass} passed, {total_fail} failed")
print("=" * 60)

sys.exit(1 if total_fail > 0 else 0)
