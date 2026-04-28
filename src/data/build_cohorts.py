"""Cohort construction for Freddie Mac loans.

Cohort key: (origination_quarter, FICO bucket, LTV bucket, loan purpose).
Output: long-form panel of monthly 90+ DPD rate per cohort.

Segmentation rationale: Project_charter.md §5.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Bucket boundaries per Project_charter.md §5. right=False semantics: [a, b).
FICO_BINS = [-float("inf"), 660, 720, 780, float("inf")]
FICO_LABELS = ["<660", "660-720", "720-780", "780+"]

LTV_BINS = [-float("inf"), 60, 80, 95, float("inf")]
LTV_LABELS = ["<60", "60-80", "80-95", "95+"]

LOAN_PURPOSE_LABELS = {"P": "Purchase", "N": "Refinance", "C": "Cash-out refi"}

DEFAULT_MIN_COHORT_SIZE = 500

_VINTAGE_RE = re.compile(r"^F(\d{2})Q(\d)")


def _parse_vintage_from_loan_seq(loan_seq: str) -> str | None:
    """Extract vintage like '2013Q1' from loan_seq_num like 'F13Q10000042'."""
    m = _VINTAGE_RE.match(loan_seq)
    if m is None:
        return None
    yy, q = m.groups()
    return f"20{yy}Q{q}"


def assign_cohort(orig_df: pd.DataFrame) -> pd.DataFrame:
    """Add cohort columns and a unique `cohort_id` to an originations frame.

    Adds: vintage, fico_bucket (label), ltv_bucket (label), loan_purpose_label,
    cohort_id. Drops loans with NA in any cohort component.

    `cohort_id` format: `<vintage>_F<fico_idx>_L<ltv_idx>_<purpose_code>`,
    e.g. `2013Q1_F2_L1_P`. Bucket codes (0-3) keep the ID filename-safe;
    the readable labels live in separate columns for plotting/tables.
    """
    df = orig_df.copy()

    df["vintage"] = df["loan_seq_num"].map(_parse_vintage_from_loan_seq)

    df["fico_bucket"] = pd.cut(
        df["credit_score"], bins=FICO_BINS, labels=FICO_LABELS, right=False,
    )
    df["ltv_bucket"] = pd.cut(
        df["original_ltv"], bins=LTV_BINS, labels=LTV_LABELS, right=False,
    )
    df["loan_purpose_label"] = df["loan_purpose"].map(LOAN_PURPOSE_LABELS)

    needed = ["vintage", "fico_bucket", "ltv_bucket", "loan_purpose_label"]
    df = df.dropna(subset=needed)

    df["cohort_id"] = (
        df["vintage"].astype(str)
        + "_F" + df["fico_bucket"].cat.codes.astype(str)
        + "_L" + df["ltv_bucket"].cat.codes.astype(str)
        + "_" + df["loan_purpose"].astype(str)
    )

    return df


def build_cohort_panel(
    orig_df: pd.DataFrame,
    perf_df: pd.DataFrame,
    min_cohort_size: int = DEFAULT_MIN_COHORT_SIZE,
) -> pd.DataFrame:
    """Aggregate loan-month performance into cohort-month 90+ DPD rates.

    Steps:
      1. Assign cohort columns to originations.
      2. Compute n_loans per cohort; drop cohorts below min_cohort_size.
      3. Inner-join to performance.
      4. Group by (cohort_id, reporting_period); compute n_active, n_serious,
         n_reconstructed, dpd_90_rate, reconstructed_share.
      5. Return long-form DataFrame sorted by (cohort_id, reporting_period).
    """
    orig = assign_cohort(orig_df)

    cohort_loan_counts = orig.groupby("cohort_id", observed=True).size()
    big_cohorts = cohort_loan_counts[cohort_loan_counts >= min_cohort_size].index
    orig = orig[orig["cohort_id"].isin(big_cohorts)]

    cohort_keys = orig[
        ["loan_seq_num", "cohort_id", "vintage",
         "fico_bucket", "ltv_bucket", "loan_purpose_label"]
    ]

    merged = perf_df.merge(cohort_keys, on="loan_seq_num", how="inner")

    grouped = (
        merged.groupby(["cohort_id", "reporting_period"], observed=True)
        .agg(
            n_active=("loan_seq_num", "size"),
            n_serious=("is_serious_delinquent", "sum"),
            n_reconstructed=("is_forbearance_reconstructed", "sum"),
        )
        .reset_index()
    )

    grouped["dpd_90_rate"] = grouped["n_serious"] / grouped["n_active"]
    grouped["reconstructed_share"] = grouped["n_reconstructed"] / grouped["n_active"]

    cohort_meta = (
        cohort_keys.drop_duplicates(subset=["cohort_id"])
        .drop(columns=["loan_seq_num"])
    )
    panel = grouped.merge(cohort_meta, on="cohort_id", how="left")

    return panel.sort_values(["cohort_id", "reporting_period"]).reset_index(drop=True)


def summarize_cohorts(panel: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-cohort summary: span, max active loans, mean/max DPD rate."""
    return (
        panel.groupby("cohort_id", observed=True)
        .agg(
            n_months=("reporting_period", "size"),
            first_period=("reporting_period", "min"),
            last_period=("reporting_period", "max"),
            max_n_active=("n_active", "max"),
            mean_dpd_90_rate=("dpd_90_rate", "mean"),
            max_dpd_90_rate=("dpd_90_rate", "max"),
        )
        .reset_index()
    )


if __name__ == "__main__":
    import sys

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from src.data.load_freddie_mac import (
        add_serious_delinquency_flag,
        load_originations,
        load_performance,
    )

    sample_dir = project_root / "data/raw/freddie_mac/sample/sample_2013"

    print("Loading Sample 2013...")
    orig = load_originations(sample_dir / "sample_orig_2013.txt")
    perf = load_performance(sample_dir / "sample_svcg_2013.txt")
    perf = add_serious_delinquency_flag(perf)
    print(f"Originations: {len(orig):,} loans")
    print(f"Performance:  {len(perf):,} loan-months\n")

    print("=== Cohort size distribution ===")
    cohorts = assign_cohort(orig)
    sizes = cohorts.groupby("cohort_id", observed=True).size()
    print(f"Total cohorts:         {len(sizes)}")
    print(f"Loans assigned:        {sizes.sum():,}")
    print(f"Loans dropped (NA):    {len(orig) - sizes.sum():,}")
    print(f"Min cohort size:       {sizes.min()}")
    print(f"Median cohort size:    {sizes.median():.0f}")
    print(f"Max cohort size:       {sizes.max():,}")
    for thresh in [50, 100, 200, 500, 1000]:
        n_kept = (sizes >= thresh).sum()
        print(f"  Cohorts ≥ {thresh:>4}:  {n_kept:>3} retained")

    print("\n=== Building cohort panel (threshold 100 for Sample) ===")
    panel = build_cohort_panel(orig, perf, min_cohort_size=100)
    print(f"Panel rows:  {len(panel):,}")
    print(f"Cohorts:     {panel['cohort_id'].nunique()}")
    print(f"Periods:     {panel['reporting_period'].min().strftime('%Y-%m')} "
          f"to {panel['reporting_period'].max().strftime('%Y-%m')}")

    print("\n=== Top 5 cohorts by max 90+ DPD rate ===")
    summary = summarize_cohorts(panel)
    print(summary.nlargest(5, "max_dpd_90_rate")[
        ["cohort_id", "max_n_active", "mean_dpd_90_rate", "max_dpd_90_rate"]
    ].to_string(index=False))

    print("\n=== Bottom 5 cohorts by max 90+ DPD rate ===")
    print(summary.nsmallest(5, "max_dpd_90_rate")[
        ["cohort_id", "max_n_active", "mean_dpd_90_rate", "max_dpd_90_rate"]
    ].to_string(index=False))

    # Trajectory snapshot for one representative cohort.
    sample_cid = panel["cohort_id"].iloc[0]
    print(f"\n=== Trajectory of {sample_cid} — first 3 months + COVID window ===")
    first = panel[panel["cohort_id"] == sample_cid]
    cols = ["reporting_period", "n_active", "n_serious",
            "dpd_90_rate", "reconstructed_share"]
    print(first.head(3)[cols].to_string(index=False))
    print("  ...")
    covid = first[(first["reporting_period"] >= "2020-03") &
                  (first["reporting_period"] <= "2020-09")]
    print(covid[cols].to_string(index=False))
