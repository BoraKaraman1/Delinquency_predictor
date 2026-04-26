# Data acquisition

Raw data is gitignored. This document describes how to populate `data/raw/`.

## Freddie Mac Single-Family Loan-Level Dataset

The primary data source.

- **Link:** [https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset)
- **Access:** free, requires account registration and license acceptance.

### Files needed

For each year 2013-2017:
- **Standard Origination data file** (one per quarter) — loan-level static features at origination.
- **Standard Monthly Performance data file** (one per quarter) — monthly observation panel through the most recent reporting period.

Place the unzipped files under `data/raw/freddie_mac/`:

```
data/raw/freddie_mac/
├── orig_2013Q1.txt
├── orig_2013Q2.txt
├── ...
├── orig_2017Q4.txt
├── perf_2013Q1.txt
├── perf_2013Q2.txt
├── ...
└── perf_2017Q4.txt
```

### Performance window

We follow performance through 2022 to capture both pre-COVID calm and COVID stress regimes. Freddie Mac's monthly performance file extends to the most recent quarter; the loader will filter to ≤ 2022-12 observations.

### Forbearance flag

Forbearance status is captured via the **Loan Forbearance Plan Indicator** (or equivalent — column name has changed across vintage releases; see the official file layout PDF). The forbearance reconstruction (treating forborne loans as 90+ DPD) depends on this column being correctly parsed.

## FRED macroeconomic data

Three series are pulled from FRED. Fetched programmatically — no manual download required.

| Variable | FRED ID | Source |
|---|---|---|
| Civilian Unemployment Rate (national, U-3) | `UNRATE` | BLS |
| FHFA House Price Index, all-transactions, USA, monthly | `HPIPONM226S` | FHFA |
| 30-Year Fixed Rate Mortgage Average | `MORTGAGE30US` | Freddie Mac PMMS |

Required Python packages: `pandas-datareader` (or direct FRED API via `fredapi`).

A FRED API key is recommended (free) for higher rate limits. Set via the `FRED_API_KEY` environment variable.

## File sizes

The full Freddie Mac monthly performance file for 2013-2017 originations is approximately [TBD] GB unzipped. After cohort construction (group-by quarterly vintage × FICO bucket × LTV bucket × loan purpose), the working panel is much smaller (a few hundred series × ~120 monthly observations).

Intermediate processed files land in `data/processed/` (also gitignored).
