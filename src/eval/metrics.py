"""Forecasting accuracy metrics and statistical significance testing.

Metrics:
  - sMAPE: symmetric mean absolute percentage error (scale-free)
  - MASE: mean absolute scaled error (scale-free, relative to naive baseline)
  - RMSE: root mean squared error (unit-anchored)

Significance test:
  - Diebold-Mariano test with HAC variance (Newey-West estimator)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric mean absolute percentage error.

    Formula: 200 * |y - ŷ| / (|y| + |ŷ|)

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        sMAPE. Returns 0.0 if both y and ŷ are zero. Returns NaN if (y + ŷ) = 0
        for a non-zero element.
    """
    numerator = np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)

    result = 200 * numerator / denominator

    result[denominator == 0.0] = 0.0

    return float(np.mean(result))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    season_period: int = 12,
) -> float:
    """Mean absolute scaled error.

    MASE = MAE / (MAE of seasonal naive forecast on training data).
    Seasonal naive: y_t = y_{t - season_period}.

    Parameters
    ----------
    y_true : np.ndarray
        True values in the test set.
    y_pred : np.ndarray
        Predicted values.
    y_train : np.ndarray
        Training series (used to compute seasonal naive baseline).
    season_period : int
        Seasonal period. Default 12 (months).

    Returns
    -------
    float
        MASE. Values < 1 indicate better-than-naive forecasts.
    """
    mae_test = np.mean(np.abs(y_true - y_pred))

    if len(y_train) < season_period:
        raise ValueError(f"Training series too short ({len(y_train)}) for season_period={season_period}")

    seasonal_naive_errors = y_train[season_period:] - y_train[:-season_period]
    mae_naive = np.mean(np.abs(seasonal_naive_errors))

    if mae_naive == 0.0:
        return float(mae_test == 0.0)

    return float(mae_test / mae_naive)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        RMSE.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int,
    loss: str = "squared",
) -> tuple[float, float]:
    """Diebold-Mariano test for forecast accuracy comparison.

    Tests H0: E[L(e_a)] = E[L(e_b)] where L is the loss function.
    Uses HAC (Newey-West) variance with lag truncation = h.

    Reference: Diebold & Mariano (1995). Comparing Predictive Accuracy.
               Journal of Business & Economic Statistics, 13(3), 253-263.

    Parameters
    ----------
    errors_a : np.ndarray
        Forecast errors for model A (shape n,).
    errors_b : np.ndarray
        Forecast errors for model B (shape n,).
    h : int
        Forecast horizon. Used to set HAC lag truncation.
    loss : str
        Loss function: "squared" (MSE) or "absolute" (MAE). Default "squared".

    Returns
    -------
    tuple[float, float]
        (dm_statistic, p_value). p_value is two-tailed.
    """
    if len(errors_a) != len(errors_b):
        raise ValueError("errors_a and errors_b must have the same length")

    if loss == "squared":
        loss_a = errors_a ** 2
        loss_b = errors_b ** 2
    elif loss == "absolute":
        loss_a = np.abs(errors_a)
        loss_b = np.abs(errors_b)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    d = loss_a - loss_b
    d_mean = np.mean(d)

    n = len(d)
    var_d = np.var(d, ddof=1)

    if var_d == 0.0:
        return 0.0, 1.0

    lag = h
    c0 = np.mean(d ** 2)
    autocovariance = 0.0

    for k in range(1, lag + 1):
        gamma_k = np.mean(d[:-k] * d[k:])
        autocovariance += 2 * gamma_k

    var_d_hac = c0 + autocovariance

    if var_d_hac <= 0.0:
        return 0.0, 1.0

    dm_stat = d_mean / np.sqrt(var_d_hac / n)
    p_value = 2 * stats.t.sf(np.abs(dm_stat), df=n - 1)

    return float(dm_stat), float(p_value)


def evaluate_fold(
    fold,
    forecasts: dict[str, pd.DataFrame],
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Evaluate forecast accuracy on a single fold.

    Parameters
    ----------
    fold : Fold
        Fold object from rolling_origin.make_folds().
    forecasts : dict[str, pd.DataFrame]
        Model predictions. Keys are model names; values are DataFrames with columns:
        cohort_id, reporting_period, y_pred.
    horizons : list[int] | None
        Forecast horizons to evaluate. Default [1, 3, 6, 12].

    Returns
    -------
    pd.DataFrame
        One row per (model, cohort, horizon). Columns: model, cohort_id, horizon,
        smape, mase, rmse, n_obs.
    """
    if horizons is None:
        horizons = [1, 3, 6, 12]

    results = []
    test_df = fold.test_df.copy()

    for model_name, pred_df in forecasts.items():
        pred_df = pred_df.copy().set_index(["cohort_id", "reporting_period"])

        for cohort_id in test_df["cohort_id"].unique():
            cohort_test = test_df[test_df["cohort_id"] == cohort_id].sort_values(
                "reporting_period"
            )
            cohort_train = fold.train_df[fold.train_df["cohort_id"] == cohort_id].sort_values(
                "reporting_period"
            )

            if len(cohort_test) == 0 or len(cohort_train) == 0:
                continue

            y_true = cohort_test["dpd_90_rate"].values
            y_train = cohort_train["dpd_90_rate"].values

            for i, h in enumerate(horizons):
                if i >= len(y_true):
                    break

                y_true_h = y_true[i : i + 1]

                try:
                    pred_val = pred_df.loc[(cohort_id, cohort_test.iloc[i]["reporting_period"]), "y_pred"]
                except (KeyError, TypeError):
                    pred_val = np.nan

                if np.isnan(pred_val):
                    continue

                y_pred_h = np.array([pred_val])

                try:
                    smape_val = smape(y_true_h, y_pred_h)
                    mase_val = mase(y_true_h, y_pred_h, y_train, season_period=12)
                    rmse_val = rmse(y_true_h, y_pred_h)
                except Exception:
                    smape_val = np.nan
                    mase_val = np.nan
                    rmse_val = np.nan

                results.append({
                    "model": model_name,
                    "cohort_id": cohort_id,
                    "horizon": h,
                    "smape": smape_val,
                    "mase": mase_val,
                    "rmse": rmse_val,
                    "n_obs": 1,
                })

    return pd.DataFrame(results) if results else pd.DataFrame(
        columns=["model", "cohort_id", "horizon", "smape", "mase", "rmse", "n_obs"]
    )


def run_dm_tests(
    metrics_df: pd.DataFrame,
    model_pairs: list[tuple[str, str]],
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Run pairwise Diebold-Mariano tests.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics from evaluate_fold(). Must contain columns: model, horizon, rmse, n_obs.
    model_pairs : list[tuple[str, str]]
        Pairs of models to compare. E.g., [("SARIMA", "ETS"), ("SARIMA", "N-BEATS")].
    horizons : list[int] | None
        Horizons to test. Default [1, 3, 6, 12].

    Returns
    -------
    pd.DataFrame
        One row per (model_pair, horizon). Columns: model_a, model_b, horizon,
        dm_stat, p_value, n_obs.
    """
    if horizons is None:
        horizons = [1, 3, 6, 12]

    results = []

    for model_a, model_b in model_pairs:
        for h in horizons:
            errors_a = metrics_df[
                (metrics_df["model"] == model_a) & (metrics_df["horizon"] == h)
            ]["rmse"].values

            errors_b = metrics_df[
                (metrics_df["model"] == model_b) & (metrics_df["horizon"] == h)
            ]["rmse"].values

            if len(errors_a) == 0 or len(errors_b) == 0:
                continue

            min_len = min(len(errors_a), len(errors_b))
            errors_a = errors_a[:min_len]
            errors_b = errors_b[:min_len]

            dm_stat, p_val = diebold_mariano_test(errors_a, errors_b, h=h, loss="squared")

            results.append({
                "model_a": model_a,
                "model_b": model_b,
                "horizon": h,
                "dm_stat": dm_stat,
                "p_value": p_val,
                "n_obs": len(errors_a),
            })

    return pd.DataFrame(results) if results else pd.DataFrame(
        columns=["model_a", "model_b", "horizon", "dm_stat", "p_value", "n_obs"]
    )


def save_results(
    metrics_df: pd.DataFrame,
    dm_df: pd.DataFrame,
    run_id: str,
    results_dir: Path = Path("results/runs"),
) -> None:
    """Save evaluation results to CSV files.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Per-(model, cohort, horizon) metrics from evaluate_fold().
    dm_df : pd.DataFrame
        DM test results from run_dm_tests().
    run_id : str
        Run identifier (e.g., timestamp). Used to create subdirectory.
    results_dir : Path
        Base results directory. Default results/runs.
    """
    run_dir = Path(results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(run_dir / "metrics.csv", index=False)
    dm_df.to_csv(run_dir / "dm_tests.csv", index=False)

    print(f"Results saved to {run_dir}/")


if __name__ == "__main__":
    print("=== metrics smoke test ===\n")

    np.random.seed(42)
    y_true = np.array([0.02, 0.025, 0.03])
    y_pred = np.array([0.02, 0.025, 0.03])
    y_train = np.array([0.01 + 0.005 * np.sin(i / 12 * 2 * np.pi) + np.random.normal(0, 0.001)
                       for i in range(24)])

    smape_val = smape(y_true, y_pred)
    print(f"sMAPE (perfect forecast): {smape_val:.4f} (expect 0.0)")
    assert np.isclose(smape_val, 0.0), "sMAPE failed for perfect forecast"

    y_pred_zero = np.array([0.0, 0.0, 0.0])
    smape_val_zero = smape(y_true, y_pred_zero)
    print(f"sMAPE (zero prediction): {smape_val_zero:.1f} (expect 200.0)")
    assert np.isclose(smape_val_zero, 200.0), "sMAPE failed for zero prediction"

    mase_val = mase(y_true, y_pred, y_train, season_period=12)
    print(f"MASE (perfect forecast): {mase_val:.4f} (expect ≈ 0.0)")
    assert mase_val < 0.1, "MASE failed for perfect forecast"

    rmse_val = rmse(y_true, y_pred)
    print(f"RMSE (perfect forecast): {rmse_val:.4f} (expect 0.0)")
    assert np.isclose(rmse_val, 0.0), "RMSE failed for perfect forecast"

    print("\n=== Diebold-Mariano test ===\n")
    errors_a = np.random.normal(0.01, 0.005, 100)
    errors_b = np.random.normal(0.01, 0.005, 100)
    dm_stat, p_val = diebold_mariano_test(errors_a, errors_b, h=1, loss="squared")
    print(f"DM stat (same distribution): {dm_stat:.4f}")
    print(f"p-value: {p_val:.4f} (expect > 0.05)")
    assert p_val > 0.05, "DM test failed for identical distributions"

    print("\nAll smoke tests passed! ✓")
