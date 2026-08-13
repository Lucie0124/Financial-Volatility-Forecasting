# Data and Feature Engineering

## Data Sources

This project uses publicly available financial and macroeconomic time series from the **Federal Reserve Economic Data (FRED)** database.

The initial dataset contains the following market indicators:

| Variable            | FRED series    | Description                           |
| ------------------- | -------------- | ------------------------------------- |
| `sp500`             | `SP500`        | S&P 500 index level                   |
| `vix`               | `VIXCLS`       | CBOE Volatility Index                 |
| `treasury_2y`       | `DGS2`         | 2-year U.S. Treasury yield            |
| `treasury_10y`      | `DGS10`        | 10-year U.S. Treasury yield           |
| `high_yield_spread` | `BAMLH0A0HYM2` | U.S. high-yield corporate bond spread |

These indicators were selected to represent complementary dimensions of market conditions:

```text
Equity market        -> S&P 500
Expected volatility  -> VIX
Short-term rates     -> 2Y Treasury yield
Long-term rates      -> 10Y Treasury yield
Credit stress        -> High-yield spread
```

The S&P 500 is the central series of the project because its daily returns are used to construct the realised-volatility forecasting target.

The VIX captures volatility implied by S&P 500 option prices, while the Treasury yields provide information about monetary-policy expectations, long-term rates, and the shape of the yield curve.

The high-yield spread was initially included to capture corporate credit-market stress.

## Data Download

The raw FRED series are downloaded using:

```text
scripts/download_fred.py
```

The script:

1. Downloads each time series.
2. Standardises the column names.
3. Saves the individual series in:

```text
data/raw/
```

4. Merges all series by date.
5. Saves the merged dataset as:

```text
data/processed/market_data.csv
```

Run the script from the repository root with:

```bash
python scripts/download_fred.py
```

## Data Cleaning

Cleaning and time-series alignment are implemented in:

```text
src/market_risk/data/prepare.py
```

The cleaning pipeline:

1. Loads `data/processed/market_data.csv`.
2. Sorts observations chronologically.
3. Uses S&P 500 trading days as the reference calendar.
4. Removes rows where the S&P 500 is unavailable.
5. Forward-fills short gaps in the other market indicators using only previously observed values.
6. Limits forward-filling to five consecutive observations.
7. Checks remaining missing values.

Forward-filling is preferred over backward-filling because backward-filling could introduce future information into earlier observations and therefore create data leakage.

## High-Yield Spread Exclusion

After cleaning, the remaining missing values were:

```text
date                    0
sp500                   0
vix                     0
treasury_10y            0
treasury_2y             0
high_yield_spread    1761
```

The large number of missing observations in `high_yield_spread` is caused by the limited historical coverage currently available for the selected FRED series.

Forward-filling these observations would not be appropriate because the early missing values have no previous observation to propagate.

Restricting the full modelling dataset to the period where high-yield spread data are available would also substantially reduce the historical sample and weaken the walk-forward forecasting experiment.

For these reasons, `high_yield_spread` is excluded from the first version of the forecasting model.

The retained raw indicators are therefore:

```text
sp500
vix
treasury_2y
treasury_10y
```

Credit-spread information may be reintroduced in a future version using a source with longer historical coverage.

## Feature Engineering

Feature engineering is implemented in:

```text
src/market_risk/features/build_features.py
```

The first version of the project intentionally uses a compact and interpretable feature set.

### Equity Returns

Daily S&P 500 log returns are defined as:

```math
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
```

The following return features are constructed:

```text
return_1d
return_5d
```

These capture short-term equity-market movements without relying directly on the non-stationary S&P 500 index level.

### Realised Volatility

Rolling realised-volatility measures are calculated over:

```text
volatility_5d
volatility_10d
volatility_20d
```

These features capture market instability over different recent horizons.

They are annualised using approximately 252 trading days per year.

Using several volatility windows helps distinguish short-lived shocks from more persistent high-volatility periods.

### VIX Features

The project includes:

```text
vix_change_1d
vix_change_5d
vix_vs_realized_vol
```

The VIX change features capture recent shifts in option-implied uncertainty.

The implied-versus-realised volatility spread is defined as:

```python
df["vix_vs_realized_vol"] = (
    df["vix"] / 100
    - df["volatility_20d"]
)
```

This measures whether option-market expectations are above or below recently realised market volatility.

### Interest-Rate Features

The slope of the yield curve is calculated as:

```python
df["yield_curve"] = (
    df["treasury_10y"]
    - df["treasury_2y"]
)
```

The project also includes:

```text
treasury_10y_change_5d
treasury_2y_change_5d
```

These features capture recent changes in long- and short-term interest-rate expectations.

## Final Engineered Feature Set

The first modelling version uses:

```text
return_1d
return_5d

volatility_5d
volatility_10d
volatility_20d

vix_change_1d
vix_change_5d
vix_vs_realized_vol

yield_curve
treasury_10y_change_5d
treasury_2y_change_5d
```

The original levels of `vix`, `treasury_2y`, and `treasury_10y` are also retained.

The feature set is deliberately kept small and economically interpretable rather than generating a large number of technical indicators.

## Missing Values After Feature Engineering

Feature engineering introduces a small number of missing observations at the beginning of the dataset:

```text
return_1d                  0.0%
return_5d                  0.2%
volatility_5d              0.2%
volatility_10d             0.4%
volatility_20d             0.8%
vix_change_1d              0.0%
vix_change_5d              0.2%
vix_vs_realized_vol        0.8%
yield_curve                0.0%
treasury_10y_change_5d     0.2%
treasury_2y_change_5d      0.2%
```

These missing values are structural rather than random.

For example:

* `return_5d` requires five previous observations.
* `volatility_20d` requires a 20-day historical window.
* `vix_change_5d` requires a VIX observation from five trading days earlier.
* `vix_vs_realized_vol` inherits the initial missing observations from `volatility_20d`.

These observations are not imputed because doing so would fabricate unavailable historical information.

Rows without sufficient history are removed only after the complete feature set and target have been constructed.
