# Market Volatility & Stress-Regime Forecasting

> **Work in progress** — a learning project. The pipeline and its evaluation are still being developed; the project is not a finished application.

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

## Data preparation

From the repository root:

```bash
python scripts/download_fred.py
python src/market_risk/data/prepare.py
```

The scripts download and merge the source series, align them to S&P 500 trading days and forward-fill short gaps (up to five observations, without using future values). Data is generated locally under `data/raw/` and `data/processed/` rather than committed to the repository.

## Project status

Modelling and evaluation are ongoing. The planned API, dashboard and deployment components are **not yet presented as completed features**.

## AI-assisted documentation

AI is used to help draft and edit `.md` files, especially to **summarise methodological choices and experimental results** as part of the learning process. The aim is to clarify and document the work, not to replace independent understanding or validation.
