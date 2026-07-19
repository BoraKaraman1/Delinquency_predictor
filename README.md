# Delinquency forecasting — classical vs modern time-series

> Experimental comparison of classical statistical and deep-learning forecasters on US mortgage delinquency cohorts, 2013-2022.

## Research question

**Primary:** On segmented US mortgage delinquency cohort data, does a globally-trained deep model outperform per-series classical forecasters, and under what conditions?

**Secondary:** Within deep models, what is the marginal value of macro covariates under realistic information conditions (2-month publication lag)?

## Data

- **Source:** [Freddie Mac Single-Family Loan-Level Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset)
- **Vintage range:** 2013-Q1 through 2017-Q4 originations
- **Performance window:** through 2022 (covers pre-COVID calm and COVID stress regimes)
- **Macro covariates:** FRED — `UNRATE`, `HPIPONM226S`, `MORTGAGE30US`. Lagged 2 months.

See [`data/README.md`](./data/README.md) for acquisition instructions.

## Cohort construction

Cohort time series defined by **origination quarter × FICO bucket × LTV bucket × loan purpose**.

| Dimension | Buckets |
|---|---|
| Origination quarter | 20 (2013-Q1 to 2017-Q4) |
| FICO at origination | <660, 660-720, 720-780, 780+ |
| LTV at origination | <60, 60-80, 80-95, 95+ |
| Loan purpose | Purchase, Refinance, Cash-out refinance |

- Minimum cohort size: 500 loans (smaller cohorts dropped — monthly rates below this threshold are sampling noise rather than signal).
- Resulting cohort count: **[NN — to fill from data]** non-empty cohorts (target ~350-500).

[Cohort size distribution / histogram placeholder.]

## Target metric

Monthly **90+ DPD rate per cohort**, with CARES-Act forbearance **reconstructed as delinquent**:

> *Any loan in active forbearance during a given month is treated as 90+ DPD for that month.*

This construction is necessary because COVID-era forbearance suppressed observed delinquency despite elevated underlying credit stress. At the May-June 2020 peak, ~7% of Freddie Mac loans were in active forbearance. Without reconstruction, the COVID period would falsely appear *less* stressed than 2018-2019 — the opposite of true credit stress.

The reconstruction is a deliberate methodological choice. Alternative constructions (observed delinquency only; exclude the forbearance window) are documented in the [project charter](./Project_charter.md).

## Models

| Camp | Model | Pooling | Covariates | Implementation |
|---|---|---|---|---|
| Classical | SARIMA (auto-ARIMA) | per-series | none | `pmdarima` |
| Classical | ETS (state-space family) | per-series | none | `statsforecast` |
| Modern | N-BEATS (Oreshkin et al. 2019) | global | none | **Hand-built from paper** |
| Modern | TFT (Lim et al. 2021) | global | 3 macro (lagged) | `pytorch-forecasting` |

The N-BEATS implementation was built without AI assistance, validated against the paper's reported sMAPE on M4 yearly (target: within 5% of paper-reported numbers). Detailed implementation choices and divergences from the reference are documented in [`src/models/nbeats/IMPLEMENTATION_NOTES.md`](./src/models/nbeats/IMPLEMENTATION_NOTES.md) (to be written alongside the implementation).

### Comparison structure

| Comparison | What's varying | Held constant | Role |
|---|---|---|---|
| (SARIMA, ETS) vs (N-BEATS, TFT) | classical vs modern | dataset, methodology | Primary finding |
| N-BEATS vs TFT | covariates | global pooling, deep architecture | Secondary analysis |

## Methodology

- **Cross-validation:** rolling-origin / expanding-window. [Fold structure: training start / step size / number of folds — to fill.]
- **Forecast horizons:** h = 1, 3, 6, 12 months.
- **Metrics:** sMAPE and MASE (primary, scale-free, comparable across cohorts). RMSE (secondary, unit-anchored).
- **Significance testing:** Diebold-Mariano test on pairwise forecast accuracy differences across cohorts and horizons.
- **Macro information rule:** uniform 2-month lag. At forecast time T₀, macros usable only up to T₀ − 2. Reflects realistic publication-lag conditions.
- **Optional reference column:** TFT with perfect macro foresight reported as an upper-bound; headline numbers use lagged macros.

## Results

### Primary finding: classical vs modern

[Headline table — sMAPE / MASE / RMSE × model × horizon, with DM-test significance markers.]

[Per-model trajectory plots on representative cohorts.]

### Per-segment breakdown

[Tables / heatmaps: per FICO band, per LTV band, per loan-purpose category.]

### Per-regime breakdown (pre-COVID vs COVID)

[Table / plot: model performance separately in 2018-2019 vs 2020-2022 folds.]

### Secondary analysis: covariate value within deep learning

[N-BEATS vs TFT (lagged macros) comparison.]

[TFT lagged macros vs TFT perfect foresight — the upper-bound reference.]

## Discussion

[*To write after experiments. Address:*]

- What the primary finding implies about classical-vs-modern in a credit-risk panel setting.
- Where deep learning helped (which horizons, segments, regimes) and where it didn't.
- Whether macro covariates added value beyond global pooling, and whether that value held up under realistic information conditions.
- How results compare to published M-competition findings and to the credit-risk forecasting literature.

## Limitations

- Single asset class (US conforming residential mortgages). Findings may not generalize to consumer credit cards, auto, or unsecured lending.
- COVID forbearance reconstruction is one defensible choice among several; alternative constructions could yield different stress-period results.
- Macro covariates are national-level; geography-specific covariates (state unemployment, regional HPI) were not included.
- Macro publication-lag rule is uniform 2-month; per-variable lags would be more realistic.
- Perfect macro foresight reported only as an upper-bound reference; headline results assume lagged macros.

## Implementation notes (N-BEATS)

The N-BEATS implementation was a deliberate hand-built component of this project, written without AI assistance from the [Oreshkin et al. (2019) paper](https://arxiv.org/abs/1905.10437). The implementation-notes document covers:

- What the paper underspecifies (initialization details, basis-function parameterization edge cases, training schedule).
- Design choices made and why.
- Where the implementation diverges from the published reference.
- Validation against paper-reported sMAPE on M4 yearly.

See [`src/models/nbeats/IMPLEMENTATION_NOTES.md`](./src/models/nbeats/IMPLEMENTATION_NOTES.md).

## Repo structure

```
.
├── README.md                # This document
├── Project_charter.md       # Decision log and rationale
├── pyproject.toml
├── data/                    # Raw and processed data (gitignored contents)
│   ├── README.md            # Data acquisition guide
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/                # Loaders, cohort construction, forbearance reconstruction
│   ├── cv/                  # Rolling-origin CV harness
│   ├── models/
│   │   ├── classical/       # SARIMA, ETS
│   │   ├── nbeats/          # Hand-built N-BEATS + IMPLEMENTATION_NOTES.md
│   │   └── tft/             # TFT wrapper around pytorch-forecasting
│   ├── eval/                # Metrics (sMAPE, MASE, RMSE), Diebold-Mariano test
│   └── tuning/              # Optuna sweeps
├── notebooks/               # Exploratory and analysis notebooks
├── results/
│   ├── tables/              # Saved CSVs of result tables
│   ├── figures/             # Plots
│   └── runs/                # Model checkpoints (gitignored)
├── paper/                   # Long-form writeup (optional, post-results)
└── tests/                   # Unit tests for CV, metrics, N-BEATS
```

## How to reproduce

[*To write once code is in place.*]

1. Clone the repo and set up the environment.
2. Acquire Freddie Mac data — see [`data/README.md`](./data/README.md).
3. Build cohort panel and macro merge.
4. Run classical models (SARIMA, ETS).
5. Run N-BEATS.
6. Run TFT.
7. Generate result tables and figures.

## References

- Box, G. E. P., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control.*
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics.*
- Hyndman, R. J., & Khandakar, Y. (2008). Automatic time series forecasting: the `forecast` package for R. *Journal of Statistical Software.*
- Hyndman, R. J., Koehler, A. B., Snyder, R. D., & Grose, S. (2002). A state space framework for automatic forecasting using exponential smoothing methods. *International Journal of Forecasting.*
- Hyndman, R. J., & Athanasopoulos, G. (2021). [*Forecasting: Principles and Practice* (3rd ed.)](https://otexts.com/fpp3/).
- Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363). *International Journal of Forecasting.*
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. *International Journal of Forecasting.*
- Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2019). [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](https://arxiv.org/abs/1905.10437). *ICLR 2020.*

## Future work

- Foundation models (Chronos, TimesFM, Lag-Llama) — zero-shot evaluation on this task.
- iTransformer (Liu et al. 2024) — cohort-as-channel framing for cross-cohort attention.
- Per-variable macro lags (UNRATE 1m, HPI 2m, MORTGAGE30US 0m) instead of uniform 2-month.
- Geographic disaggregation (state- or MSA-level cohorts).
- Other asset classes (autos, cards) — does the classical-vs-modern finding generalize?
