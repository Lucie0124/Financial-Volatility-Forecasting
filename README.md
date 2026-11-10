# Market Volatility & Stress-Regime Forecasting

> **Work in progress** — a learning project. Progress has been on hold since September 2026, but I plan to resume work and continue improving the pipeline and its evaluation; the project is not a finished application.

> **Correction after review:** the forward target and persistence now use the same annualised five-session RMS definition. Data, results and models have been recalculated. Ridge's pooled MAE gain is about 4.77%, rather than the earlier 38%.

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

Both measures use the shared definition in `src/market_risk/features/volatility.py` and are expressed as annualised decimal volatilities. The rolling sample-standard-deviation features remain separate predictors. See [target definition](docs/02_target_definition.md).

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


## Critical review with Codex

I developed the initial project, modelling approach and exploratory notebooks. I recently used Codex to challenge the methodology. This review identified errors in target normalisation, benchmark consistency and evaluation; these were corrected with Codex’s assistance while preserving the original workflow.

To reproduce the corrected results after installing the project, run the existing modules from the repository root:

```bash
python -m pip install -e ".[dev]"
python -m market_risk.data.prepare
python -m market_risk.features.build_features
python -m market_risk.features.build_target
python -m market_risk.evaluation.walk_forward
python -m market_risk.modelling.ridge
python -m market_risk.modelling.lightgbm_model
python -m market_risk.evaluation.regimes
python -m market_risk.training.train_final_models
python -m pytest -q
```

Then run each notebook from top to bottom in a fresh kernel, from the `notebooks/` directory. Results depend on package versions; no untouched holdout, statistical significance or economic backtest is claimed.
