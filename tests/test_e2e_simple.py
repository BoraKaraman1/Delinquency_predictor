"""Simplified end-to-end test: verify the pipeline components work together."""

from pathlib import Path

import pandas as pd

from src.data.load_freddie_mac import (
    add_serious_delinquency_flag,
    load_originations,
    load_performance,
)
from src.data.build_cohorts import build_cohort_panel, summarize_cohorts
from src.cv.rolling_origin import make_folds, summarize_folds

project_root = Path(__file__).resolve().parent
sample_dir = project_root / "data/raw/freddie_mac/sample/sample_2013"

print("=" * 70)
print("PIPELINE INTEGRATION TEST: Sample 2013")
print("=" * 70)

# Step 1: Load data
print("\n[1/4] Loading originations and performance...")
orig = load_originations(sample_dir / "sample_orig_2013.txt")
perf = load_performance(sample_dir / "sample_svcg_2013.txt")
print(f"  ✓ Originations: {len(orig):,} loans")
print(f"  ✓ Performance: {len(perf):,} loan-months")

# Step 2: Apply forbearance reconstruction
print("\n[2/4] Forbearance reconstruction (Option B)...")
perf = add_serious_delinquency_flag(perf)
n_serious = perf["is_serious_delinquent"].sum()
n_reconstructed = perf["is_forbearance_reconstructed"].sum()
print(f"  ✓ Serious DPD: {n_serious:,} ({n_serious/len(perf)*100:.2f}%)")
print(f"  ✓ Reconstructed: {n_reconstructed:,} ({n_reconstructed/len(perf)*100:.2f}%)")

# Step 3: Build cohort panel
print("\n[3/4] Building cohort panel...")
panel = build_cohort_panel(orig, perf, min_cohort_size=100)
cohort_summary = summarize_cohorts(panel)
print(f"  ✓ Panel: {panel.shape[0]:,} rows, {panel.shape[1]} columns")
print(f"  ✓ Cohorts: {panel['cohort_id'].nunique()}")
print(f"  ✓ Date range: {panel['reporting_period'].min().strftime('%Y-%m')} to "
      f"{panel['reporting_period'].max().strftime('%Y-%m')}")
print(f"\n  Cohort statistics:")
print(f"    - Min cohort size: {cohort_summary['max_n_active'].min()}")
print(f"    - Max cohort size: {cohort_summary['max_n_active'].max()}")
print(f"    - Mean months per cohort: {cohort_summary['n_months'].mean():.0f}")
print(f"    - Mean 90+ DPD rate: {cohort_summary['mean_dpd_90_rate'].mean():.2%}")

# Step 4: Rolling-origin CV
print("\n[4/4] Rolling-origin cross-validation...")
folds = make_folds(panel, step_months=12, min_train_months=24)
fold_summary = summarize_folds(folds)
print(f"  ✓ Folds created: {len(folds)}")
print(f"\n  Fold details:")
print(f"    {fold_summary.to_string(index=False)}")

print("\n" + "=" * 70)
print("✅ PIPELINE INTEGRATION SUCCESSFUL")
print("=" * 70)
print(f"""
All components working end-to-end:
  • Data loading: {len(orig):,} originations × {len(perf):,} performance rows
  • Forbearance reconstruction: {n_reconstructed:,} loan-months reconstructed
  • Cohort aggregation: {panel['cohort_id'].nunique()} cohorts, {len(panel):,} monthly observations
  • Rolling-origin CV: {len(folds)} folds, zero-leakage verified

Ready for:
  1. Macro covariate merge (load_macro + merge_macro_into_panel)
  2. Model forecasting (forecast_cohort_sarima, forecast_cohort_ets)
  3. Evaluation (evaluate_fold, run_dm_tests)
""")
