# Market Volatility & Stress-Regime Forecasting

> **Work in progress** — a learning project. The pipeline and its evaluation are still being developed; the project is not a finished application.

> **Historical results under review:** an audit identified a missing division by five in the forward variance and a persistence benchmark based on a different estimator. The source definitions are now corrected and tested. The checked-in modelling dataset, results, notebook outputs and model artifacts still describe the previous experiment and must be regenerated before reporting performance for the corrected method. The modelling-data loader rejects labels that do not match the corrected definition.

This project explores forecasting **S&P 500 realised volatility over the next five trading days** and identifying **normal, elevated-volatility and stress regimes**. The goal is reproducible market-risk analysis, not price prediction or investment advice.

## Data and methodology

- **Data:** Daily S&P 500 (`SP500`), VIX (`VIXCLS`) and US Treasury yields (`DGS2`, `DGS10`) from [FRED](https://fred.stlouisfed.org/). High-yield credit spreads were investigated but excluded from the initial modelling dataset due to insufficient historical coverage.
- **Features:** Historical returns, realised volatility, VIX movements, Treasury yields and yield-curve slope.
- **Models:** Volatility-persistence baseline, Ridge regression and LightGBM.
- **Evaluation:** Walk-forward validation, forecast errors and analysis by volatility regime, with particular attention to avoiding future-data leakage.

The target is the annualised volatility computed from the **five subsequent daily log returns**:

$$
RV_{t,t+5} = \sqrt{\frac{252}{5}\sum_{i=1}^{5}r_{t+i}^{2}},
\qquad r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
$$

Persistence applies the same non-centred estimator to the five most recent returns, including the return at the forecast date:

$$
\widehat{RV}_{t,t+5}^{\mathrm{persistence}}
= \sqrt{\frac{252}{5}\sum_{j=0}^{4}r_{t-j}^{2}}.
$$

Both measures use the shared definition in `src/market_risk/features/volatility.py` and are expressed as annualised decimal volatilities. The rolling sample-standard-deviation features remain separate predictors. See [target definition and migration status](docs/02_target_definition.md).

## Data preparation

From the repository root:

```bash
python scripts/download_fred.py
python src/market_risk/data/prepare.py
```

The scripts download and merge the source series, align them to S&P 500 trading days and forward-fill short gaps (up to five observations, without using future values). Raw downloads under `data/raw/` are ignored by Git; the current CSV snapshot under `data/processed/` is versioned.

## Project status

Modelling and evaluation are ongoing. The planned API, dashboard and deployment components are **not yet presented as completed features**.

## AI-assisted documentation

AI is used to help draft and edit `.md` files in 'docs' folder, especially to **summarise methodological choices and experimental results** as part of the learning process. The aim is to clarify and document the work, not to replace independent understanding or validation.
