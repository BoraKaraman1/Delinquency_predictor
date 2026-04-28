"""Loaders for the Freddie Mac Single-Family Loan-Level Dataset.

Both Sample and Standard tiers share the same 32-column schema for
originations and the same 32-column schema for monthly servicing/performance.

Field positions are documented in the Freddie Mac Single-Family Loan-Level
Dataset General User Guide. Schema verified against Sample 2013 files
(see data/README.md).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ORIG_COLUMNS = [
    "credit_score",                     # 1
    "first_payment_date",               # 2 — YYYYMM
    "first_time_homebuyer_flag",        # 3
    "maturity_date",                    # 4 — YYYYMM
    "msa",                              # 5
    "mortgage_insurance_pct",           # 6
    "number_of_units",                  # 7
    "occupancy_status",                 # 8
    "original_cltv",                    # 9
    "original_dti",                     # 10
    "original_upb",                     # 11
    "original_ltv",                     # 12
    "original_interest_rate",           # 13
    "channel",                          # 14
    "ppm_flag",                         # 15
    "amortization_type",                # 16
    "property_state",                   # 17
    "property_type",                    # 18
    "postal_code",                      # 19
    "loan_seq_num",                     # 20 — JOIN KEY
    "loan_purpose",                     # 21 — P / N / C
    "original_loan_term",               # 22
    "number_of_borrowers",              # 23
    "seller_name",                      # 24
    "servicer_name",                    # 25
    "super_conforming_flag",            # 26
    "pre_harp_loan_seq_num",            # 27
    "program_indicator",                # 28
    "harp_indicator",                   # 29
    "property_valuation_method",        # 30
    "interest_only_indicator",          # 31
    "mi_cancellation_indicator",        # 32
]

PERF_COLUMNS = [
    "loan_seq_num",                          # 1 — JOIN KEY
    "monthly_reporting_period",              # 2 — YYYYMM
    "current_actual_upb",                    # 3
    "current_loan_delinquency_status",       # 4 — '0','1','2','3',...,'RA','XX'
    "loan_age",                              # 5
    "remaining_months_to_legal_maturity",    # 6
    "defect_settlement_date",                # 7
    "modification_flag",                     # 8
    "zero_balance_code",                     # 9
    "zero_balance_effective_date",           # 10
    "current_interest_rate",                 # 11
    "current_deferred_upb",                  # 12
    "ddlpi",                                 # 13 — due date of last paid installment
    "mi_recoveries",                         # 14
    "net_sales_proceeds",                    # 15
    "non_mi_recoveries",                     # 16
    "total_expenses",                        # 17
    "legal_costs",                           # 18
    "maintenance_and_preservation_costs",    # 19
    "taxes_and_insurance",                   # 20
    "miscellaneous_expenses",                # 21
    "actual_loss_calculation",               # 22
    "modification_cost",                     # 23
    "step_modification_flag",                # 24
    "deferred_payment_plan",                 # 25
    "estimated_loan_to_value",               # 26
    "zero_balance_removal_upb",              # 27
    "delinquent_accrued_interest",           # 28
    "delinquency_due_to_disaster",           # 29
    "borrower_assistance_status_code",       # 30 — F / R / T / blank
    "current_month_modification_cost",       # 31
    "interest_bearing_upb",                  # 32
]

# FICO sentinel: '9999' encodes "score not available".
_FICO_UNKNOWN = 9999
# Original LTV / CLTV / DTI sentinels: '999' encodes unknown.
_PCT_UNKNOWN = 999


def load_originations(path: str | Path) -> pd.DataFrame:
    """Read a Freddie Mac SF originations PSV file.

    Returns one row per loan, keyed on `loan_seq_num`. Numeric fields are
    converted; sentinel values for unknowns (9999 for FICO, 999 for LTV/CLTV/DTI)
    are converted to NA.
    """
    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=ORIG_COLUMNS,
        dtype=str,
        engine="c",
        na_values=[""],
        low_memory=False,
    )

    df["credit_score"] = pd.to_numeric(df["credit_score"], errors="coerce").astype("Int64")
    df.loc[df["credit_score"] == _FICO_UNKNOWN, "credit_score"] = pd.NA

    for col in ("original_ltv", "original_cltv", "original_dti"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        df.loc[df[col] == _PCT_UNKNOWN, col] = pd.NA

    df["original_upb"] = pd.to_numeric(df["original_upb"], errors="coerce").astype("Int64")
    df["original_interest_rate"] = pd.to_numeric(df["original_interest_rate"], errors="coerce")
    df["original_loan_term"] = pd.to_numeric(df["original_loan_term"], errors="coerce").astype("Int64")
    df["number_of_borrowers"] = pd.to_numeric(df["number_of_borrowers"], errors="coerce").astype("Int64")

    df["first_payment_date"] = pd.to_datetime(df["first_payment_date"], format="%Y%m", errors="coerce")
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], format="%Y%m", errors="coerce")

    return df


def load_performance(
    path: str | Path,
    end_date: str = "2022-12",
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Read a Freddie Mac SF monthly performance PSV file in chunks.

    Filters to reporting periods ≤ `end_date` (inclusive) before concatenating
    chunks, so the in-memory DataFrame omits post-window rows.

    Returns a long-form frame: one row per (loan_seq_num, monthly_reporting_period).
    """
    end_yyyymm = int(end_date.replace("-", "")[:6])

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        sep="|",
        header=None,
        names=PERF_COLUMNS,
        dtype=str,
        engine="c",
        na_values=[""],
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk["monthly_reporting_period"] = pd.to_numeric(chunk["monthly_reporting_period"])
        chunk = chunk[chunk["monthly_reporting_period"] <= end_yyyymm]
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)

    # Type conversions on the filtered frame.
    df["current_actual_upb"] = pd.to_numeric(df["current_actual_upb"], errors="coerce")
    df["loan_age"] = pd.to_numeric(df["loan_age"], errors="coerce").astype("Int64")
    df["remaining_months_to_legal_maturity"] = pd.to_numeric(
        df["remaining_months_to_legal_maturity"], errors="coerce"
    ).astype("Int64")
    df["current_interest_rate"] = pd.to_numeric(df["current_interest_rate"], errors="coerce")

    # YYYYMM → datetime for joins / time-series indexing.
    df["reporting_period"] = pd.to_datetime(
        df["monthly_reporting_period"], format="%Y%m", errors="coerce"
    )

    return df


def add_serious_delinquency_flag(perf_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the Option B forbearance reconstruction (Project_charter.md §6).

    Adds two boolean columns:

      * `is_serious_delinquent` — TRUE if the loan-month is 90+ DPD by the
        observed delinquency status OR (per the reconstruction rule) is in
        active CARES-Act forbearance. This is the headline target signal.

      * `is_forbearance_reconstructed` — TRUE only for loan-months that
        qualify *exclusively* via the forbearance arm of the OR. Useful for
        sensitivity analysis (how much of the COVID-era serious-DPD signal
        is policy-mediated rather than observed?).

    Delinquency-status decoding:
      * '0','1','2'           → not 90+ DPD (current, 30 DPD, 60 DPD)
      * '3' and any longer    → 90+ DPD (3 months, 4 months, ...)
      * 'RA'                  → REO acquired (terminal — implies serious DPD)
      * 'XX' / blank          → unknown, treated as not 90+ DPD
    """
    df = perf_df.copy()

    dlq_raw = df["current_loan_delinquency_status"].fillna("").astype(str).str.strip()
    dlq_numeric = pd.to_numeric(dlq_raw, errors="coerce")
    is_dlq_90plus = (dlq_numeric >= 3) | (dlq_raw == "RA")

    in_forbearance = (
        df["borrower_assistance_status_code"].fillna("").astype(str).str.strip() == "F"
    )

    df["is_serious_delinquent"] = is_dlq_90plus | in_forbearance
    df["is_forbearance_reconstructed"] = in_forbearance & ~is_dlq_90plus

    return df


if __name__ == "__main__":
    # Smoke test on Sample 2013.
    project_root = Path(__file__).resolve().parents[2]
    sample_dir = project_root / "data/raw/freddie_mac/sample/sample_2013"

    print(f"Project root: {project_root}")
    print(f"Sample dir:   {sample_dir}\n")

    print("=== load_originations ===")
    orig = load_originations(sample_dir / "sample_orig_2013.txt")
    print(f"Rows:        {len(orig):,}")
    print(f"Columns:     {len(orig.columns)}")
    print(f"FICO range:  {orig['credit_score'].min()} – {orig['credit_score'].max()} "
          f"(median {orig['credit_score'].median():.0f}, NA {orig['credit_score'].isna().sum()})")
    print(f"LTV range:   {orig['original_ltv'].min()} – {orig['original_ltv'].max()} "
          f"(median {orig['original_ltv'].median():.0f}, NA {orig['original_ltv'].isna().sum()})")
    print(f"Loan purpose distribution:\n{orig['loan_purpose'].value_counts(dropna=False).to_string()}")

    print("\n=== load_performance (filtered to ≤ 2022-12) ===")
    perf = load_performance(sample_dir / "sample_svcg_2013.txt")
    print(f"Rows:                  {len(perf):,}")
    print(f"Reporting period range: {perf['reporting_period'].min().strftime('%Y-%m')} "
          f"to {perf['reporting_period'].max().strftime('%Y-%m')}")
    print(f"Unique loans observed: {perf['loan_seq_num'].nunique():,}")

    print("\n=== add_serious_delinquency_flag ===")
    perf = add_serious_delinquency_flag(perf)
    n = len(perf)
    n_serious = int(perf["is_serious_delinquent"].sum())
    n_reconstructed = int(perf["is_forbearance_reconstructed"].sum())
    print(f"Serious DPD loan-months:        {n_serious:>10,} ({n_serious / n * 100:.3f}%)")
    print(f"  via observed DLQ status:      {n_serious - n_reconstructed:>10,}")
    print(f"  via forbearance reconstruct.: {n_reconstructed:>10,} ({n_reconstructed / n * 100:.3f}%)")
