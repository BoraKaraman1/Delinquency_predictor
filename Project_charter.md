# Project charter — Delinquency forecasting (classical vs modern time-series)

## 1. Goals and framing

**Decision:** This is a portfolio piece with two specific, named goals: strengthening MSc applications and demonstrating knowledge of **time-series methods specifically** (not general ML or production engineering).

**Why:** The predecessor project (a credit-scoring pipeline with Supabase / FastAPI / medallion architecture) already demonstrated production engineering. Repeating that shape would be redundant. This project must look *different* — experimental, methodological, with a single defensible finding rather than a feature list.

**What this rules out:**
- Production infrastructure (Airflow, scoring endpoints, medallion architecture).
- LightGBM as the protagonist — it would dominate via feature engineering and obscure the time-series story. It's at most a context baseline.
- A sprawling "I tried 10 things" project. Two camps, four well-tuned models.

## 2. Core thesis

**Decision:** A direct comparison of classical time-series methods against modern (deep-learning) time-series methods, framed as a many-related-series forecasting problem (M4/M5-style).

**Why:** A comparison-with-finding is the project shape that distinguishes you from candidates who "build a forecaster." The MSc panel rewards "what did you learn?" not "what did you ship?"

**Headline question:** *On segmented mortgage delinquency cohort data, does a globally-trained deep model outperform per-series classical forecasters, and under what conditions?*

**Narrative shape:** **single primary finding** (does globally-trained deep learning beat per-series classical?) with **one secondary analysis** (marginal value of covariates within deep learning, under realistic information conditions). Not two parallel findings.

## 3. Dataset

**Decision:** Freddie Mac Single-Family Loan-Level dataset.

**Why over alternatives:**
- *Lending Club:* origination data, weak panel structure, overused in bootcamp portfolios.
- *Home Credit (Kaggle):* panel-ish but classification-framed, harder to defend as a time-series project.
- *Synthetic:* undermines the "real finding" framing.
- *Fannie Mae:* a sibling of Freddie Mac — pick one, not both.
- *M5 (Walmart retail):* considered and rejected. The classical-vs-modern question is already answered on M5 (LightGBM-based ensembles won; pure deep learning underperformed). Walking into a settled debate weakens the project.

Freddie Mac wins because it has real monthly performance panel data per loan, is public, well-documented, and has an explicit DPD status field. Trade-off: mortgages are slow-default 30-year products, very different from cards. The writeup must be honest about that.

## 4. Time window

**Decision:** 2013-2017 origination vintages, performance followed through 2022.

**Why:**
- 2013-2017 is post-Dodd-Frank, so underwriting standards are stable and comparable across vintages.
- Avoids GFC-era confounding (the 2007-2009 originations were a different world).
- Following through 2022 captures both pre-COVID calm (2018-2019) and COVID stress (2020-2022) — gives a contrast for per-regime breakdowns.

**What this enables:** The interesting kind of finding — "deep models win during stress, classical wins during calm" or whatever the truth turns out to be. Conditional findings are what readers remember.

## 5. Cohort segmentation

**Decision:** Cohort time series defined by **origination quarter × FICO bucket × LTV bucket × loan purpose**.

| Dimension | Buckets |
|---|---|
| Origination quarter | 20 (2013-Q1 to 2017-Q4) |
| FICO at origination | <660, 660-720, 720-780, 780+ |
| LTV at origination | <60, 60-80, 80-95, 95+ |
| Loan purpose | Purchase, Refinance, Cash-out refinance |

Theoretical max: 20 × 4 × 4 × 3 = 960 combinations. Realistic non-empty: **~350-500 cohorts**.

**Why this scheme:**
- Origination quarter × FICO × LTV is the canonical **vintage analysis** framework that real risk teams use → connects to your prior credit-risk work, signals domain credibility.
- Loan purpose was added as a fourth dimension (over the original three-axis design) to expand the cohort count from ~150-250 to ~350-500 without thinning per-cohort sample size. More series gives the global deep models more pooling signal.
- Loan purpose is a **named credit-risk dimension** (refis and purchases have meaningfully different default patterns) — defensible domain choice, not arbitrary cohort-padding.
- Interpretable cohorts ("2015-Q3 originations, FICO 660-720, LTV 80-95, refinance") that examiners can picture immediately.

**Rejected alternatives:**
- *Lower the per-cohort minimum threshold instead of adding a dimension:* would expand cohort count by accepting noisier per-cohort rates. Confounds the model comparison; small-sample binomial noise hurts classical methods more than deep, biasing the result.
- *Adding state (50x more cohorts):* unworkable.
- *Random clustering / quantile binning:* less interpretable, no domain credibility.

**Methodology hygiene:**
- Drop cohorts with fewer than **500 loans** (below this, monthly rates are sampling noise, not signal).
- Optionally cap large cohorts at ~5,000 loans (random subsample) for storage.

## 6. Target metric and the COVID complication

**Decision:** Monthly **90+ DPD rate per cohort**, with CARES-Act forbearance **reconstructed as delinquency** (Option B).

**Why 90+ DPD specifically:** It's the standard "serious delinquency" cutoff in mortgage credit risk. Lower thresholds (30+) have more events but less severity signal. Higher thresholds (180+) are too rare in a 5-year window.

**Why reconstruct forbearance:** During COVID, the CARES Act let federally-backed mortgages enter forbearance — payments paused, *not counted as delinquent*. At peak (mid-2020), ~7% of Freddie Mac loans were forborne. If you used observed delinquency, the COVID stress period would look *less* stressed than 2018-2019 — the opposite of true credit stress.

**The reconstruction rule:** any loan in active forbearance is treated as 90+ DPD for that month.

**Why this matters for the writeup:** Most portfolio projects don't engage with policy-mediated outcomes. Naming this issue and constructing a defensible target shows methodological sophistication — exactly the signal an MSc panel looks for. There will be a methodology section in the writeup explaining the construction.

**Rejected alternatives:**
- Option A (observed only): the COVID stress period loses its bite.
- Option C (exclude forbearance window): throws away the contrast we deliberately included by extending through 2022.

## 7. Model lineup

**Decision:** Four models in two camps.

| Camp | Model | Pooling | Covariates | Role |
|---|---|---|---|---|
| Classical | SARIMA (auto-ARIMA per series) | per-series | none | Co-protagonist |
| Classical | ETS (Hyndman state-space family) | per-series | none | Co-protagonist |
| Modern | N-BEATS (Oreshkin et al. 2019) | global | none | Secondary deep — pure pooling effect |
| Modern | TFT (Lim et al. 2021) | global | 3 macro (lagged) | Primary deep — full architecture |

**Why this specific four:**
- **SARIMA** is the canonical Box-Jenkins method, recognized in any MSc curriculum. Auto-ARIMA handles order selection.
- **ETS** is treated as a full **co-protagonist with equal tuning depth** to SARIMA (auto-ETS over the full Hyndman state-space family with AIC selection and residual diagnostics — *not* a default `ets()` call). Hyndman is more bullish on ETS than ARIMA on many datasets; the M3/M4 results reflect that.
- **N-BEATS** is pure feedforward, no covariates, won M4 tracks — the canonical "deep model that demonstrably works on forecasting." Also the hand-built component (see §11).
- **TFT** is the strongest interpretable deep forecaster: handles static + known-future + observed-past covariates natively, exposes attention weights / variable importance for the writeup.

**Comparison structure (one primary finding + one secondary analysis):**

| Comparison | What's varying | What's held constant | Role |
|---|---|---|---|
| (SARIMA, ETS) → (N-BEATS, TFT) | classical vs modern | dataset, methodology | Primary finding |
| N-BEATS → TFT | covariates | global pooling, deep architecture | Secondary analysis |

**Rejected alternatives:**
- *DeepAR:* a defensible lower-ceiling pick if TFT proves too fiddly. Held in reserve as a fallback.
- *iTransformer (2024):* too novel/exotic for a secondary, and its strength (cross-variable attention) doesn't align with the project's panel framing. Mentioned in related-work and future-work sections only — signals frontier awareness without committing to evaluating it.
- *LightGBM:* would obscure the time-series story.
- *Prophet:* not really classical-statistical, mixed reception in the academic community.
- *Foundation models (Chronos, TimesFM, Lag-Llama):* mentioned in future work, not evaluated.

## 8. Macro covariates (TFT only)

**Decision:** Three macro covariates from FRED, all treated as observed-past, with a uniform 2-month lag.

| Variable | FRED ID | Mechanism |
|---|---|---|
| Unemployment rate (national, U-3) | `UNRATE` | Income-loss → missed payment |
| FHFA HPI (national, monthly) | `HPIPONM226S` | Collateral → underwater → strategic default |
| 30Y fixed mortgage rate | `MORTGAGE30US` | Refi / competing risks → sample composition |

**Why three, and why these:** Each has a *named, defensible mechanism* — that's what makes it MSc-grade rather than "I dumped 50 features in." Three is enough for a story without overfitting or drowning the writeup.

**Information rule: uniform 2-month lag.** At any forecast time T₀, the model can use macro values up to T₀ − 2, inclusive. Same lag for all three.

**Why uniform 2-month lag (not perfect foresight, not per-variable lags):**
- Matches the slowest-publishing macro (FHFA HPI ~2 month lag) — never optimistic.
- Simpler to defend in the writeup than per-variable lags (1m / 2m / 0m).
- Reflects realistic information availability — what a production model would actually have.
- Removes the "perfect foresight" caveat that would otherwise need disclaiming.

**What this means for TFT:** macros remain *observed-past* inputs, but with a 2-month wall. The model sees their history up to T₀ − 2 and nothing for T₀ − 1, T₀, or any of the forecast horizon. At long horizons, macros effectively become a static last-known-value anchor.

**Optional reference point:** also run TFT with perfect foresight as an upper-bound column in the results table. Headline number is the lagged version. Adds one extra TFT training run, much stronger writeup.

**Rejected alternatives:**
- *Perfect foresight:* unrealistic, requires a "limitation" disclaimer; relegated to optional reference column.
- *Forecast macros separately:* introduces another error source, confounds the architecture comparison.
- *Per-variable lags:* slightly more realistic but adds complexity for marginal gain. Could ablate at end if time allows.
- 10Y Treasury (redundant with mortgage rate), CPI (weaker mortgage-specific signal), VIX/sentiment (soft mechanisms). Could be added in v2.

## 9. Methodology

**Decision:** A specific stack of evaluation choices that an MSc panel will recognize.

- **Rolling-origin / expanding-window cross-validation**, not random splits. Non-negotiable for time-series.
- **Forecasting metrics:** sMAPE and MASE primary (scale-free, comparable across series). RMSE as a secondary unit-anchored metric.
- **Statistical significance:** Diebold-Mariano test on pairwise forecast accuracy differences. Most portfolio projects skip this — including it lifts the project's tier.
- **Per-horizon breakdown:** report h = 1, 3, 6, 12 months separately. Classical typically wins short-horizon, deep wins long-horizon — that nuance is the finding.
- **Per-segment breakdown:** per FICO band, per LTV band, per loan-purpose category, per regime (pre-COVID vs COVID). The conditional findings.
- **Cite the original papers:** Box-Jenkins, Hyndman & Khandakar (auto-ARIMA), Hyndman et al. (ETS state-space framework), Oreshkin et al. (N-BEATS), Lim et al. (TFT), Makridakis (M-competitions), Diebold-Mariano.

## 10. Compute

**Decision:** Base M5 MacBook Pro with PyTorch MPS backend. Colab as a fallback for stubborn MPS issues only.

**Why:** Dataset is small by deep-learning standards (~350-500 series × ~120 months ≈ 50-60k observations). The constraint isn't compute, it's experimental rigor. M5 is overkill for the modeling and right-sized for the methodology.

**Practical note:** PyTorch ≥ 2.5, MPS gradient bugs in earlier versions are mostly fixed. PyTorch Forecasting library for the TFT implementation; **N-BEATS is hand-built, not from the library** (see §11).

## 11. Fully hand-built component (no AI assistance)

**Decision:** **N-BEATS implementation from the Oreshkin et al. (2019) paper.**

**Why this specifically:**
- N-BEATS from scratch is real engineering and real paper-reading. Architecture: basis stacks (generic vs interpretable trend/seasonality), doubly-residual stacking, backcast/forecast split, the loss. ~300-500 lines of clean PyTorch.
- Implementing a deep model from scratch is much more impressive than implementing a CV loop. The CV loop is methodology hygiene; N-BEATS from scratch is depth signal.
- Forces engagement with what the paper underspecifies — design choices become explicit, defensible.
- The contrast in the writeup ("here's what I built without AI vs here's what I built with") is genuinely valuable interview signal.

**Required to make the hand-build credible:**
- Build it from the paper, not from `pytorch-forecasting`'s implementation. Read [their code](https://github.com/sktime/pytorch-forecasting) only *after* yours works.
- Validate against a reference implementation on a toy dataset — within 5% of reported sMAPE on M4 yearly
- Write a short **"implementation notes" doc** covering: what the paper underspecifies, design choices made, where you diverged from the reference. This document is the artifact of the hand-build effort — it's what an MSc panel will actually read.

**Rejected alternatives for hand-build component:**
- *Rolling-origin CV harness:* the original choice — N-BEATS is a stronger depth signal.
- *Implementing TFT from scratch:* too ambitious, would consume the project.
- *Implementing the loss function:* too small a slice.
- *Implementing data loading:* not differentiating.

## 12. What's next

In sequence:
1. **README skeleton** — write the headline finding placeholder and section structure before any code. If you can't articulate the finding now, the project is unfocused.
2. **Repo structure** — lean: `data/`, `src/`, `notebooks/`, `results/`, `paper/` (or similar). No medallion architecture.
3. **Data acquisition** — confirm Freddie Mac account access works.
4. **Rolling-origin CV harness** — first code module, since everything downstream depends on it.
5. **Classical models on small slice** — get SARIMA + ETS running end-to-end on one cohort before scaling.
6. **N-BEATS hand-build** — implement from the paper, validate on toy data, write the implementation-notes doc.
7. **TFT** — last, after the evaluation harness is solid and N-BEATS is working.
8. **Hyperparameter sweep** — Optuna preferred (simpler than Ray Tune for this scale).
9. **Writeup** — refine throughout, not at the end.
