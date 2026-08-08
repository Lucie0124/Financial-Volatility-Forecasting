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

