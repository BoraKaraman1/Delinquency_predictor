"""Rolling-origin (expanding-window) cross-validation for time-series panel data.

Yields fold objects containing train/test splits with a global cutoff date.
Respects the time axis within each cohort series — no cross-series leakage.
Designed for both per-series (SARIMA, ETS) and global (N-BEATS, TFT) models.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Fold:
    """One fold from rolling-origin CV.

    Attributes
    ----------
    fold_idx : int
        Fold index (0-based).
    cutoff : pd.Timestamp
        Global cutoff date. train_df covers ≤ cutoff; test_df covers > cutoff.
    train_df : pd.DataFrame
        Panel subset with reporting_period <= cutoff. Used for model training.
    test_df : pd.DataFrame
        Panel subset with cutoff < reporting_period <= cutoff + 12m.
        Used for evaluation at h = 1, 3, 6, 12 months ahead.
    """
    fold_idx: int
    cutoff: pd.Timestamp
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def make_folds(
    panel: pd.DataFrame,
    step_months: int = 6,
    min_train_months: int = 24,
    max_horizon: int = 12,
) -> list[Fold]:
    """Generate rolling-origin CV folds from a cohort panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Cohort panel from build_cohort_panel(), sorted by (cohort_id, reporting_period).
        Must have columns: cohort_id, reporting_period, dpd_90_rate.
    step_months : int
        Number of months to advance between consecutive fold cutoffs.
        Default 6 → ~12 folds on 108 months.
    min_train_months : int
        Minimum training window length in months. Skip fold if training < min_train_months.
    max_horizon : int
        Maximum forecast horizon in months (test window length). Default 12.

    Returns
    -------
    list[Fold]
        List of Fold objects. Each fold has expanding training window and 12-month test window.
        Zero-leakage invariant: no row in train_df appears in test_df (by time).
    """
    panel = panel.copy().sort_values(["cohort_id", "reporting_period"]).reset_index(drop=True)

    date_range = pd.date_range(
        start=panel["reporting_period"].min(),
        end=panel["reporting_period"].max(),
        freq="MS",  # month start
    )

    if len(date_range) < min_train_months + max_horizon:
        raise ValueError(
            f"Panel spans only {len(date_range)} months; need >= "
            f"{min_train_months + max_horizon} (min_train + max_horizon)"
        )

    folds = []
    fold_idx = 0

    for cutoff in date_range[min_train_months::step_months]:
        test_end = cutoff + pd.DateOffset(months=max_horizon)

        if test_end > date_range[-1]:
            break

        train_df = panel[panel["reporting_period"] <= cutoff].copy()
        test_df = panel[
            (panel["reporting_period"] > cutoff) & (panel["reporting_period"] <= test_end)
        ].copy()

        if len(test_df) == 0:
            continue

        folds.append(
            Fold(
                fold_idx=fold_idx,
                cutoff=cutoff,
                train_df=train_df,
                test_df=test_df,
            )
        )
        fold_idx += 1

    return folds


def summarize_folds(folds: list[Fold]) -> pd.DataFrame:
    """Create a diagnostic summary of fold statistics.

    Parameters
    ----------
    folds : list[Fold]
        List of Fold objects from make_folds().

    Returns
    -------
    pd.DataFrame
        One row per fold: fold_idx, cutoff, n_train_rows, n_test_rows,
        n_train_cohorts, n_test_cohorts.
    """
    summaries = []
    for fold in folds:
        summaries.append({
            "fold_idx": fold.fold_idx,
            "cutoff": fold.cutoff,
            "n_train_rows": len(fold.train_df),
            "n_test_rows": len(fold.test_df),
            "n_train_cohorts": fold.train_df["cohort_id"].nunique(),
            "n_test_cohorts": fold.test_df["cohort_id"].nunique(),
        })
    return pd.DataFrame(summaries)


if __name__ == "__main__":
    print("=== rolling_origin smoke test ===\n")

    # Create synthetic 3-cohort × 48-month panel.
    dates = pd.date_range("2019-01", periods=48, freq="MS")
    cohorts = ["2013Q1_F0_L1_P", "2013Q1_F1_L2_N", "2013Q2_F2_L3_C"]
    rows = []
    for cohort in cohorts:
        for date in dates:
            rows.append({
                "cohort_id": cohort,
                "reporting_period": date,
                "dpd_90_rate": 0.02 + 0.005 * (date.month % 3),
            })

    synthetic_panel = pd.DataFrame(rows)
    print(f"Synthetic panel: {synthetic_panel.shape}")
    print(f"Date range: {synthetic_panel['reporting_period'].min()} "
          f"to {synthetic_panel['reporting_period'].max()}\n")

    # Generate folds.
    folds = make_folds(synthetic_panel, step_months=6, min_train_months=24)
    print(f"Number of folds: {len(folds)}\n")

    # Print fold summary.
    summary = summarize_folds(folds)
    print("Fold summary:")
    print(summary)

    # Zero-leakage invariant check.
    print("\n=== Zero-leakage verification ===\n")
    for fold in folds:
        train_periods = set(fold.train_df["reporting_period"])
        test_periods = set(fold.test_df["reporting_period"])
        overlap = train_periods & test_periods
        assert len(overlap) == 0, f"Fold {fold.fold_idx}: leakage detected! Overlap: {overlap}"
    print("All folds pass zero-leakage check ✓")

    # Temporal order check.
    print("\n=== Temporal order verification ===\n")
    for fold in folds:
        assert (fold.train_df["reporting_period"] <= fold.cutoff).all(), \
            f"Fold {fold.fold_idx}: train_df contains dates > cutoff"
        assert (fold.test_df["reporting_period"] > fold.cutoff).all(), \
            f"Fold {fold.fold_idx}: test_df contains dates <= cutoff"
    print("All folds pass temporal order check ✓")
