"""FRED macro covariate loader with 2-month lag for realistic information availability.

Three series from FRED (via fredapi):
- UNRATE: unemployment rate (national, U-3)
- HPIPONM226S: FHFA HPI all-transactions national monthly
- MORTGAGE30US: 30Y fixed mortgage rate

Information rule: uniform 2-month lag. At any forecast origin T₀, the model observes
macro values up to T₀ − 2 (inclusive). Lag is applied by shifting the FRED DatetimeIndex
forward so a value at 2020-03 becomes observable at 2020-05 with a 2-month lag.

FRED API key (optional): env var FRED_API_KEY. If not set, fredapi falls back to
the .fredapikey config file or unauthenticated requests (lower rate limits).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fredapi import Fred


def load_macro(
    start: str = "2012-01",
    end: str = "2023-12",
    lag_months: int = 2,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch FRED macro series and apply uniform lag.

    Parameters
    ----------
    start : str
        Start date in "YYYY-MM" format.
    end : str
        End date in "YYYY-MM" format.
    lag_months : int
        Uniform lag in months. A FRED value dated 2020-03 becomes observable at
        2020-03 + lag_months = 2020-05.
    api_key : str | None
        FRED API key. If None, reads from FRED_API_KEY env var or .fredapikey file.

    Returns
    -------
    pd.DataFrame
        Index: DatetimeIndex (monthly start-of-month). Columns: UNRATE, HPIPONM226S,
        MORTGAGE30US. Index values represent the reporting_period at which each macro
        observation becomes observable (already shifted by lag_months).
    """
    if api_key is None:
        api_key = os.environ.get("FRED_API_KEY")

    fred = Fred(api_key=api_key)

    unrate = fred.get_series("UNRATE", observation_start=start, observation_end=end)
    hpi = fred.get_series("HPIPONM226S", observation_start=start, observation_end=end)
    mortgage = fred.get_series("MORTGAGE30US", observation_start=start, observation_end=end)

    macro = pd.DataFrame({
        "UNRATE": unrate,
        "HPIPONM226S": hpi,
        "MORTGAGE30US": mortgage,
    })

    # Shift the index forward by lag_months.
    # A value at 2020-03 becomes observable at 2020-03 + 2 months = 2020-05.
    macro.index = macro.index + pd.DateOffset(months=lag_months)

    return macro


def merge_macro_into_panel(
    panel: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join macro covariates onto the cohort panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Cohort panel with reporting_period column (datetime).
    macro : pd.DataFrame
        Macro DataFrame from load_macro(), indexed by DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Panel with three new columns: UNRATE, HPIPONM226S, MORTGAGE30US.
        Rows before macro data is available → NaN (intentional).
    """
    result = panel.copy()
    result = result.merge(
        macro,
        left_on="reporting_period",
        right_index=True,
        how="left",
    )
    return result


if __name__ == "__main__":
    print("=== load_macro smoke test ===\n")

    if os.environ.get("FRED_API_KEY"):
        macro = load_macro(start="2019-01", end="2022-12", lag_months=2)
        print(f"Shape: {macro.shape}")
        print(f"Date range: {macro.index.min()} to {macro.index.max()}")
        print(f"Earliest observable date: {macro.index.min()}")
        print(f"Expected (start + lag_months): 2019-01 + 2 months = 2019-03")
        assert macro.index.min() >= pd.Timestamp("2019-03"), "Lag not applied correctly"
        print(f"\nFirst 5 rows:\n{macro.head()}")
        print(f"\nLast 5 rows:\n{macro.tail()}")
    else:
        print("FRED_API_KEY env var not set. Creating synthetic macro DataFrame for merge test.")
        macro = pd.DataFrame({
            "UNRATE": [3.5, 3.6, 3.7],
            "HPIPONM226S": [180.0, 185.0, 190.0],
            "MORTGAGE30US": [6.5, 6.6, 6.7],
        }, index=pd.date_range("2019-03", periods=3, freq="MS"))

    print("\n=== merge_macro_into_panel smoke test ===\n")
    synthetic_panel = pd.DataFrame({
        "cohort_id": ["2013Q1_F0_L1_P"] * 3,
        "reporting_period": pd.date_range("2019-03", periods=3, freq="MS"),
        "dpd_90_rate": [0.02, 0.025, 0.03],
    })
    print(f"Synthetic panel:\n{synthetic_panel}\n")

    merged = merge_macro_into_panel(synthetic_panel, macro)
    print(f"Merged panel (with macro):\n{merged}\n")
    assert "UNRATE" in merged.columns, "UNRATE not merged"
    assert "HPIPONM226S" in merged.columns, "HPIPONM226S not merged"
    assert "MORTGAGE30US" in merged.columns, "MORTGAGE30US not merged"
    print("Merge successful!")
