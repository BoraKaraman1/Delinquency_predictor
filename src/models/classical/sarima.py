"""SARIMA forecaster with automatic order selection via auto_arima.

Per-series model: each cohort gets its own SARIMA fit.
Multi-step forecasting: fits once, forecasts 12 steps ahead, returns h = [1, 3, 6, 12].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima


def fit_predict_sarima(
    series: pd.Series,
    horizons: list[int] | None = None,
) -> dict[str, float]:
    """Fit SARIMA to a single time series and forecast multi-step ahead.

    Parameters
    ----------
    series : pd.Series
        Univariate time series (e.g., dpd_90_rate for one cohort).
        Index should be time-sorted (ascending).
    horizons : list[int] | None
        Forecast horizons. Default [1, 3, 6, 12].

    Returns
    -------
    dict[str, float]
        Keys: "h1", "h3", "h6", "h12". Values: point forecasts at each horizon.
        Returns NaN for any horizon if the fit fails.
    """
    if horizons is None:
        horizons = [1, 3, 6, 12]

    try:
        auto_arima_model = auto_arima(
            series,
            seasonal=True,
            m=12,
            stepwise=True,
            information_criterion="aic",
            max_p=3,
            max_q=3,
            max_P=2,
            max_Q=2,
            d=None,
            D=None,
            error_action="ignore",
            suppress_warnings=True,
        )

        if auto_arima_model is None or not hasattr(auto_arima_model, "predict"):
            return {f"h{h}": np.nan for h in horizons}

        forecasts = auto_arima_model.predict(n_periods=max(horizons))

        result = {}
        for h in horizons:
            h_idx = h - 1
            if h_idx < len(forecasts):
                result[f"h{h}"] = float(forecasts[h_idx])
            else:
                result[f"h{h}"] = np.nan

        return result

    except Exception:
        return {f"h{h}": np.nan for h in horizons}


def forecast_cohort_sarima(
    cohort_id: str,
    train_df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Forecast one cohort's dpd_90_rate using SARIMA.

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
        Returns NaN for all horizons if extraction fails.
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

        forecasts = fit_predict_sarima(series, horizons=horizons)

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
    print("=== SARIMA smoke test ===\n")

    np.random.seed(42)
    t = np.arange(48)
    synthetic_series = pd.Series(
        0.02 + 0.005 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.002, 48),
        index=pd.date_range("2019-01", periods=48, freq="MS"),
    )
    synthetic_series = synthetic_series.clip(lower=0.0)

    print(f"Synthetic series: {len(synthetic_series)} months")
    print(f"Range: {synthetic_series.min():.4f} to {synthetic_series.max():.4f}\n")

    print("Fitting SARIMA...")
    forecasts = fit_predict_sarima(synthetic_series)
    print(f"Forecasts: {forecasts}\n")

    assert "h1" in forecasts, "h1 missing"
    assert "h3" in forecasts, "h3 missing"
    assert "h6" in forecasts, "h6 missing"
    assert "h12" in forecasts, "h12 missing"

    for h, val in forecasts.items():
        if not np.isnan(val):
            assert 0.0 <= val <= 1.0, f"{h} = {val} out of [0, 1] range"

    print("forecast_cohort_sarima test:")
    synthetic_panel = pd.DataFrame({
        "cohort_id": ["2013Q1_F0_L1_P"] * 48,
        "reporting_period": pd.date_range("2019-01", periods=48, freq="MS"),
        "dpd_90_rate": synthetic_series.values,
    })

    result = forecast_cohort_sarima("2013Q1_F0_L1_P", synthetic_panel)
    print(result)

    assert len(result) == 1, "Expected 1 row"
    assert "cohort_id" in result.columns, "cohort_id missing"
    print("\nSARIMA smoke test passed! ✓")
