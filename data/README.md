# Data acquisition

Raw data is gitignored. This document describes how to populate `data/raw/`.

## Strategy: Sample first, then Standard

Freddie Mac offers two free tiers behind the same login:

| Tier | Size | Purpose |
|---|---|---|
| Sample | ~1% of loans per vintage | Development, schema validation |
| Standard | Full dataset (all loans) | Production results |

**Pull Sample first.** Get the loader and forbearance reconstruction working on Sample, *then* commit to the Standard download. Standard for our window (2013-2017) is ~30-60 GB unzipped — verifying the pipeline on Sample first prevents costly re-downloads.

## Step 1: Register and authenticate

1. Navigate to the [Freddie Mac SF Loan-Level Dataset landing page](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset).
2. Click through to the data portal at `freddiemac.embs.com`.
3. Register a free account; accept the data license.
4. Approval is typically instant.

## Step 2: Download the Sample dataset

From the portal, download **Single-Family Loan-Level Dataset Sample**. You receive one zip containing all vintages. Each vintage folder contains two pipe-separated `.txt` files:

- `sample_orig_<YYYYQQ>.txt` — originations (one row per loan, ~25 fields)
- `sample_svcg_<YYYYQQ>.txt` — monthly servicing/performance (one row per loan-month, ~30 fields)

Place under `data/raw/freddie_mac/sample/`.

## Step 3: Read the file layout PDF

Required, not optional. The portal links a document titled **Single-Family Loan-Level Dataset General User Guide**, which contains the column-by-column schema. PSV files have **no header row**, so reading them requires an explicit column list. Field order has changed across releases — always use the layout PDF that matches the release year of the files you downloaded.

## Step 4: Validate on Sample 2013-Q1

Before downloading Standard, confirm three things on the Sample data for one quarter:

1. **Join works:** origination and servicing files merge cleanly on `Loan Sequence Number`.
2. **Delinquency coding parses:** the current loan delinquency status field has values like `0`, `1`, `2`, `3`, ..., `RA`, etc. as documented.
3. **Forbearance flag is populated:** the `Borrower Assistance Status Code` field is non-empty for at least some loans during 2020-2022 months.

If any check fails — especially #3 — stop and verify you have the right release of the data files. The forbearance reconstruction (treating forborne loans as 90+ DPD) is critical to this project's methodology.

## Step 5: Download the Standard dataset for 2013-2017

Once Sample is validated, return to the portal and download **Single-Family Loan-Level Dataset (Standard)**. The portal now supports **per-year downloads** (one click per year instead of per quarter), so the window 2013-2017 is **5 annual archives**, named `historical_data_<YYYY>.zip`.

### Where to drop the downloaded zips

Drop incoming yearly zips directly into:

```
data/raw/freddie_mac/standard/_yearly_zips/
```

The leading underscore marks this as a temporary landing zone — extracted data lands one level up in per-quarter folders (see structure below).

### Archive structure (verified)

Each yearly zip contains 4 quarterly zips:

```
historical_data_2015.zip
├── historical_data_2015Q1.zip       (~370 MB)
├── historical_data_2015Q2.zip       (~430 MB)
├── historical_data_2015Q3.zip       (~360 MB)
└── historical_data_2015Q4.zip       (~330 MB)
```

Each quarterly zip in turn contains the two PSV files:

```
historical_data_2015Q1.zip
├── historical_data_2015Q1.txt        (originations)
└── historical_data_time_2015Q1.txt   (monthly performance)
```

### Disk usage estimates

| Stage | Size (5 years total) |
|---|---|
| Yearly zips (compressed) | ~7 GB |
| Quarterly zips (compressed) | ~8 GB |
| Unzipped PSV files | **~40-50 GB** |

The unzipped txt files are what the loader reads; quarterly zips can be deleted after extraction. Yearly zips can be retained as a re-extraction safety net.

### Target directory structure

```
data/raw/freddie_mac/
├── sample/                                # 1% Sample data (validation set)
│   ├── sample_2013/
│   │   ├── sample_orig_2013.txt
│   │   └── sample_svcg_2013.txt
│   └── sample_<other>/
└── standard/
    ├── _yearly_zips/                      # Incoming yearly zips drop here
    │   ├── historical_data_2013.zip
    │   ├── historical_data_2014.zip
    │   ├── historical_data_2015.zip       ← already in place
    │   ├── historical_data_2016.zip
    │   └── historical_data_2017.zip
    ├── 2013Q1/
    │   ├── historical_data_2013Q1.txt
    │   └── historical_data_time_2013Q1.txt
    ├── 2013Q2/
    ├── ...
    └── 2017Q4/
```

### Extraction (run once all 5 yearly zips have arrived)

From the project root:

```bash
cd data/raw/freddie_mac/standard

# 1. Yearly zips → quarterly zips (kept inside _yearly_zips/)
for z in _yearly_zips/historical_data_*.zip; do
  unzip -o "$z" -d _yearly_zips/
done

# 2. Quarterly zips → per-quarter folders with PSV files
for qz in _yearly_zips/historical_data_????Q?.zip; do
  q=$(basename "$qz" .zip)
  quarter=${q#historical_data_}
  mkdir -p "$quarter"
  unzip -o "$qz" -d "$quarter/"
done

# 3. (Optional) free ~8 GB by deleting the quarterly zips after extraction
rm _yearly_zips/historical_data_????Q?.zip
```

There is no public API. Download is manual via browser session. Plan disk space before starting (~40-50 GB unzipped for the full window).

## Performance window

We follow performance through **2022-12** to capture pre-COVID calm and COVID stress regimes. Freddie Mac's monthly performance file extends through the most recent reporting quarter — the loader filters to ≤ 2022-12.

## Forbearance flag — where to find it

The forbearance reconstruction depends on this field being correctly parsed:

- **File:** monthly servicing/performance file
- **Field:** `Borrower Assistance Status Code` (sometimes `Borrower Assistance Plan` in older layouts)
- **Values:** `F` = Forbearance, `R` = Repayment plan, `T` = Trial period (modification), blank/space = no assistance plan
- **Reconstruction rule:** any month where this code is `F` is treated as 90+ DPD for that loan-month.

This field was added to the Standard dataset in a 2017 release and applies retroactively to older files. If your download lacks the field, you have an older release — re-pull from the portal.

## FRED macroeconomic data

Three series are pulled from FRED programmatically — no manual download.

| Variable | FRED ID | Source |
|---|---|---|
| Civilian Unemployment Rate (national, U-3) | `UNRATE` | BLS |
| FHFA House Price Index, all-transactions, USA, monthly | `HPIPONM226S` | FHFA |
| 30-Year Fixed Rate Mortgage Average | `MORTGAGE30US` | Freddie Mac PMMS |

Required Python packages: `pandas-datareader` (or direct FRED API via `fredapi`).

A FRED API key is recommended (free, registered) for higher rate limits. Set via the `FRED_API_KEY` environment variable.

## Reading large files

The Standard servicing file for one quarter contains ~10 years × every loan × monthly observations — multiple GB per file. Read in chunks (`pandas.read_csv(..., chunksize=...)`) and aggregate to cohort level before holding anything in memory. The cohort-level panel is small (~few hundred series × ~120 monthly observations); only the raw input is large.
