"""Exponential Smoothing (ETS) forecaster with full Hyndman state-space family and AIC selection.

Per-series model: each cohort gets its own ETS fit.
Model selection: fits all 30 valid configurations (2 errors × 5 trend/damping × 3 seasonality),
selects the winner by minimum AIC.
Multi-step forecasting: returns h = [1, 3, 6, 12].
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


def fit_predict_ets(
    series: pd.Series,
    horizons: list[int] | None = None,
    season_period: int = 12,
) -> dict[str, float]:
    """Fit ETS (full Hyndman state-space) to a single time series and forecast multi-step ahead.

    Fits all valid configurations from the 30-configuration Hyndman taxonomy:
    - Error: additive, multiplicative
    - Trend: additive, multiplicative, none (each with/without damping)
    - Seasonality: additive, multiplicative, none

    Skips multiplicative configurations for series with minimum <= 0.
    Selects the model with minimum AIC.

    Parameters
    ----------
    series : pd.Series
        Univariate time series (e.g., dpd_90_rate for one cohort).
        Index should be time-sorted (ascending).
    horizons : list[int] | None
        Forecast horizons. Default [1, 3, 6, 12].
    season_period : int
        Seasonal period. Default 12 (months).

    Returns
    -------
    dict[str, float]
        Keys: "h1", "h3", "h6", "h12". Values: point forecasts at each horizon.
        Returns NaN for any horizon if all fits fail.
    """
    if horizons is None:
        horizons = [1, 3, 6, 12]

    error_types = ["add", "mul"]
    trend_types = ["add", "mul", None]
    seasonal_types = ["add", "mul", None]
    damping_options = [False, True]

    best_model = None
    best_aic = np.inf

    skip_mul = series.min() <= 0.0

    for error, trend, seasonal in itertools.product(
        error_types, trend_types, seasonal_types
    ):
        damping_opts = [False] if trend is None else damping_options

        for damped in damping_opts:
            if skip_mul and (error == "mul" or trend == "mul" or seasonal == "mul"):
                continue

            try:
                model = ETSModel(
                    series,
                    error_type=error,
                    trend=trend,
                    seasonal=seasonal,
                    damped_trend=damped,
                    seasonal_periods=season_period,
                    initialization_method="estimated",
                )
                fitted = model.fit(disp=False)

                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_model = fitted

            except Exception:
                continue

    if best_model is None:
        return {f"h{h}": np.nan for h in horizons}

    try:
        forecasts = best_model.forecast(steps=max(horizons))

        result = {}
        for h in horizons:
            h_idx = h - 1
            if h_idx < len(forecasts):
                result[f"h{h}"] = float(forecasts.iloc[h_idx])
            else:
                result[f"h{h}"] = np.nan

        return result

    except Exception:
        return {f"h{h}": np.nan for h in horizons}


def forecast_cohort_ets(
    cohort_id: str,
    train_df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Forecast one cohort's dpd_90_rate using ETS.

    Parameters
    ----------
    cohort_id : str
        Cohort identifier to extract from train_df.
    train_df : pd.DataFrame
        Panel subset with columns: cohort_id, reporting_period, dpd_90_rate.
        Sorted by reporting_period.
    horizons : list[int] | None
        Forecast horizons. Default [1, 3, 6, 12].

    Returns
    -------
    pd.DataFrame
        One row with columns: cohort_id, h1, h3, h6, h12 (point forecasts).
        Returns NaN for all horizons if extraction or fitting fails.
    """
    if horizons is None:
        horizons = [1, 3, 6, 12]

    try:
        cohort_data = train_df[train_df["cohort_id"] == cohort_id].sort_values(
            "reporting_period"
        )

        if len(cohort_data) == 0:
            return pd.DataFrame(
                {
                    "cohort_id": [cohort_id],
                    **{f"h{h}": [np.nan] for h in horizons},
                }
            )

        series = cohort_data["dpd_90_rate"].reset_index(drop=True)

        forecasts = fit_predict_ets(series, horizons=horizons, season_period=12)

        row = {"cohort_id": cohort_id}
        row.update(forecasts)

        return pd.DataFrame([row])

    except Exception:
        return pd.DataFrame(
            {
                "cohort_id": [cohort_id],
                **{f"h{h}": [np.nan] for h in horizons},
            }
        )


if __name__ == "__main__":
    print("=== ETS smoke test ===\n")

    np.random.seed(42)
    t = np.arange(48)
    synthetic_series = pd.Series(
        0.02 + 0.005 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.002, 48),
        index=pd.date_range("2019-01", periods=48, freq="MS"),
    )
    synthetic_series = synthetic_series.clip(lower=0.0)

    print(f"Synthetic series: {len(synthetic_series)} months")
    print(f"Range: {synthetic_series.min():.4f} to {synthetic_series.max():.4f}\n")

    print("Fitting ETS (full Hyndman family with AIC selection)...")
    forecasts = fit_predict_ets(synthetic_series)
    print(f"Forecasts: {forecasts}\n")

    assert "h1" in forecasts, "h1 missing"
    assert "h3" in forecasts, "h3 missing"
    assert "h6" in forecasts, "h6 missing"
    assert "h12" in forecasts, "h12 missing"

    print("\nforecast_cohort_ets test:")
    synthetic_panel = pd.DataFrame({
        "cohort_id": ["2013Q1_F0_L1_P"] * 48,
        "reporting_period": pd.date_range("2019-01", periods=48, freq="MS"),
        "dpd_90_rate": synthetic_series.values,
    })

    result = forecast_cohort_ets("2013Q1_F0_L1_P", synthetic_panel)
    print(result)

    assert len(result) == 1, "Expected 1 row"
    assert "cohort_id" in result.columns, "cohort_id missing"
    print("\nETS smoke test passed! ✓")
