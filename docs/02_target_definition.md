# Forecasting Target

## Objective

The project forecasts **S&P 500 realised volatility over the next five trading days**.

Unlike stock-price or return forecasting, the objective is not to predict market direction.

Instead, the model estimates the expected magnitude of future market fluctuations.

## Daily Returns

Let (P_t) denote the S&P 500 closing level on trading day (t).

Daily log returns are defined as:

```math
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
```

## Five-Day Forward Realised Volatility

For a forecast produced at the end of trading day (t), the target uses the five subsequent daily returns:

```math
r_{t+1},\; r_{t+2},\; r_{t+3},\; r_{t+4},\; r_{t+5}
```

The five-day forward realised-volatility target is defined as:

```math
RV^{(5)}_t =
\sqrt{
\frac{252}{5}
\sum_{i=1}^{5} r_{t+i}^{2}
}
```

where:

* (r_{t+i}) is the S&P 500 log return on future trading day (t+i);
* squared returns measure the magnitude of market movements regardless of direction;
* dividing by 5 produces the average daily squared return over the forecast window;
* multiplying by 252 annualises the variance using approximately 252 trading days per year;
* taking the square root converts variance back into volatility.

A target value of:

```text
0.20
```

therefore corresponds to approximately:

```text
20% annualised realised volatility
```

over the five-day forecast window.

## Why Annualise Volatility?

Annualisation places volatility on a standard scale that is easier to interpret and compare across different horizons.

It also makes the target directly comparable in units with market volatility measures such as the VIX, which is quoted in annualised volatility terms.

Annualisation changes only the scale of the target. It does not change the underlying forecasting problem.

## Temporal Alignment

The project maintains a strict distinction between information available at prediction time and information used to construct the target:

```text
Past and present                           Future

t-19 ... t-2  t-1   t   |   t+1  t+2  t+3  t+4  t+5
─────────────────────────|────────────────────────────
       Features          |       Target returns
                         |
 information available   |    information to predict
    at prediction time   |
```

Features use only information observed on or before day (t).

The target uses only returns observed after day (t).

This separation is essential to prevent look-ahead bias and data leakage.

The final few observations in the dataset necessarily have missing target values because five future trading days are unavailable. These rows are removed rather than imputed.

## Why Forecast Realised Volatility?

Volatility is a fundamental quantity in financial risk management.

Unlike expected returns, which are often highly noisy and difficult to predict, volatility tends to exhibit persistent temporal structure.

A well-known empirical property is **volatility clustering**: high-volatility periods tend to be followed by other high-volatility periods, while calm periods also tend to persist.

This makes future realised volatility a meaningful forecasting target.

Realised-volatility forecasting is widely studied in financial econometrics and is relevant to:

* portfolio risk monitoring;
* Value-at-Risk estimation;
* volatility targeting;
* derivatives analysis;
* position sizing;
* stress monitoring;
* risk-aware portfolio allocation.

## Why a Five-Day Horizon?

Five trading days correspond approximately to one market week.

This horizon provides a compromise between:

```text
1-day horizon
-> very noisy

5-day horizon
-> short-term but more stable

20+ day horizon
-> smoother but slower to react to market regime changes
```

A five-day forecast remains responsive to rapidly changing market conditions while being less noisy than a single-day volatility target.

## Relation to Realised-Volatility Literature

The target is motivated by the realised-volatility forecasting literature.

Research by Andersen, Bollerslev, Diebold, and Labys established realised volatility as a useful ex-post measure of market variability and demonstrated substantial persistence in volatility dynamics.

The HAR-RV framework introduced by Corsi also models volatility across multiple horizons, motivating the use of recent volatility features at different time scales.

This project follows the same general forecasting idea while using daily close-to-close returns rather than high-frequency intraday returns.

The target should therefore be interpreted as:

> five-day realised volatility estimated from daily S&P 500 close-to-close returns

rather than a high-frequency realised-volatility estimator.

## References

* Andersen, T. G., Bollerslev, T., Diebold, F. X. & Labys, P. — *Modeling and Forecasting Realized Volatility*
* Corsi, F. — *A Simple Approximate Long-Memory Model of Realized Volatility*
* Cboe / S&P Dow Jones Indices — VIX methodology and interpretation
