# Market Volatility & Stress-Regime Forecasting

Financial markets can shift rapidly from stable conditions to periods of elevated uncertainty. Identifying these changes early is important for portfolio monitoring, risk management and investment decision-making.

This project develops an end-to-end machine-learning pipeline to forecast **S&P 500 realised volatility over the next five trading days** using publicly available market, interest-rate and credit-risk indicators. The input data include recent equity returns, the VIX, Treasury yields, the yield-curve slope and high-yield credit spreads.

The project compares a simple volatility-persistence baseline with regularised linear models and LightGBM. Because financial time series are highly dependent on time, the models are evaluated using **walk-forward validation** rather than random train-test splitting, ensuring that predictions are generated using past information only.

In addition to forecasting volatility, the system classifies market conditions into **normal, elevated-volatility and stress regimes**. Performance is evaluated not only through standard regression metrics, but also through the model’s ability to identify periods of market stress.

The final application is designed to provide:

* five-day-ahead volatility forecasts;
* market stress-regime classification;
* explanations of the indicators driving each prediction;
* historical forecast and error analysis;
* predictions served through a FastAPI API;
* an interactive Streamlit dashboard;
* reproducible training, automated tests and containerised deployment.

The objective is not to predict asset prices or provide investment recommendations. Instead, the project explores how machine learning can support transparent and reproducible **market-risk monitoring**.

riskpulse/
├── README.md
├── pyproject.toml
├── Dockerfile
├── configs/
│   └── series.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   └── riskpulse/
│       ├── data/
│       │   ├── fred_client.py
│       │   ├── align.py
│       │   └── validation.py
│       ├── features/
│       │   ├── returns.py
│       │   ├── volatility.py
│       │   └── macro.py
│       ├── modelling/
│       │   ├── baselines.py
│       │   ├── train.py
│       │   ├── walk_forward.py
│       │   └── regimes.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── stress_analysis.py
│       │   └── plots.py
│       ├── explainability/
│       │   └── explanations.py
│       ├── api/
│       │   ├── main.py
│       │   └── schemas.py
│       └── ui/
│           └── app.py
├── tests/
└── .github/
    └── workflows/
        └── ci.yml


## Dataset

This project uses publicly available financial and macroeconomic time series from the **Federal Reserve Economic Data (FRED)** database.

The current dataset contains five daily indicators:

| FRED series    | Project column      | Description                           |
| -------------- | ------------------- | ------------------------------------- |
| `SP500`        | `sp500`             | S&P 500 index level                   |
| `VIXCLS`       | `vix`               | CBOE Volatility Index                 |
| `DGS10`        | `treasury_10y`      | 10-year U.S. Treasury yield           |
| `DGS2`         | `treasury_2y`       | 2-year U.S. Treasury yield            |
| `BAMLH0A0HYM2` | `high_yield_spread` | U.S. high-yield corporate bond spread |

These series provide information about equity-market performance, implied volatility, interest rates, the yield curve, and credit-market stress.

The data are not committed directly to the repository. Instead, the project provides reproducible scripts to download and prepare them locally.

---

## Downloading the Data

The script:

```text
scripts/download_fred.py
```

downloads each FRED time series, standardizes its column name, and saves the individual files in:

```text
data/raw/
```

It then merges all series by date and creates:

```text
data/processed/market_data.csv
```

The resulting dataset contains the following columns:

```text
date
sp500
vix
treasury_10y
treasury_2y
high_yield_spread
```

To download the data, run the following command from the repository root:

```bash
python scripts/download_fred.py
```

After execution, the repository should contain:

```text
Financial-Volatility-Forecasting/
├── data/
│   ├── raw/
│   │   ├── SP500.csv
│   │   ├── VIXCLS.csv
│   │   ├── DGS10.csv
│   │   ├── DGS2.csv
│   │   └── BAMLH0A0HYM2.csv
│   └── processed/
│       └── market_data.csv
```

The script also reports the number and percentage of missing observations for each variable after the merge.

---

## Data Preparation

The data-cleaning logic is implemented in:

```text
src/market_risk/data/prepare.py
```

This step transforms the merged dataset into a cleaner time series suitable for feature engineering and modelling.

The preparation pipeline currently:

1. Loads `data/processed/market_data.csv`.
2. Sorts observations chronologically.
3. Uses S&P 500 trading days as the reference calendar by removing rows where `sp500` is unavailable.
4. Forward-fills short gaps in the other market indicators using previously observed values only.
5. Limits forward-filling to five consecutive observations to avoid carrying stale values across long missing periods.
6. Checks the remaining missing values.
7. Saves the cleaned dataset for the next stage of the pipeline.

The indicator columns currently processed are:

```text
vix
treasury_10y
treasury_2y
high_yield_spread
```

Forward-filling is used instead of backward-filling because backward-filling could introduce future information into earlier observations, creating **data leakage**.

To run the preparation step from the repository root:

```bash
python src/market_risk/data/prepare.py
```

The cleaned dataset is saved as:

```text
data/processed/market_data_clean.csv
```

This cleaned dataset is then used for feature engineering, including market returns, realised-volatility measures, interest-rate and credit-spread features, and the five-day forward volatility forecasting target.



### Missing Values After Cleaning

After aligning the series to S&P 500 trading days and forward-filling short gaps, the remaining missing values were:

```text
date                    0
sp500                   0
vix                     0
treasury_10y            0
treasury_2y             0
high_yield_spread    1761
```

The large number of missing observations in `high_yield_spread` is not caused by the cleaning pipeline. The FRED series `BAMLH0A0HYM2` currently provides only a limited recent history, while the rest of the dataset extends much further back.

Because most of the early observations are unavailable, forward-filling would not be appropriate: there is no previous value to propagate, and using later observations would introduce future information and create data leakage.

Restricting the entire dataset to the period where high-yield spreads are available would also reduce the amount of historical data too much for a robust walk-forward forecasting experiment.

For these reasons, `high_yield_spread` is excluded from the first version of the modelling dataset.

The retained indicators are therefore:

* `sp500`
* `vix`
* `treasury_10y`
* `treasury_2y`

This keeps a longer historical sample while preserving information about equity performance, implied volatility, interest rates, and the yield curve. Credit-spread data may be reintroduced in a later version of the project using a data source with a longer historical record.
