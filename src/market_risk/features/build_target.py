"""Build the 5-day forward realised volatility target variable for the S&P 500 index"""

from pathlib import Path
import pandas as pd
import numpy as np

from market_risk.features.volatility import (
    VOLATILITY_WINDOW,
    annualized_realized_volatility,
)

PATH_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PATH_ROOT / "data" / "processed" / "market_features.csv"
OUTPUT_PATH = PATH_ROOT / "data" / "processed" / "model_dataset.csv"
TARGET_COLUMN = "target_volatility_5d"


def build_target(df):
    """Attach annualised RMS volatility of returns t+1 through t+5."""
    df = df.copy()

    # Exclude the current return, then align the complete forward window
    # with the date at which the prediction is made.
    df[TARGET_COLUMN] = annualized_realized_volatility(
        df["return_1d"].shift(-1)
    ).shift(-(VOLATILITY_WINDOW - 1))
    
    return df


def validate_target(df):
    """Reject stored labels that do not match the current target definition.

    Call this on the complete chronological dataset before splitting it;
    a truncated fold does not contain every return needed to check its labels.
    """
    expected = build_target(df)[TARGET_COLUMN]
    if not np.allclose(
        df[TARGET_COLUMN].to_numpy(),
        expected.to_numpy(),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    ):
        raise ValueError(
            "Stored target values do not match the annualised five-session "
            "RMS definition sqrt((252 / 5) * sum(future_returns**2)). "
            "Regenerate model_dataset.csv with "
            "`python -m market_risk.features.build_target` before evaluation "
            "or training. Historical results and model artifacts must also "
            "be regenerated before reporting corrected performance."
        )


def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df = build_target(df)
    
    print(df[["date", "return_1d", "volatility_5d", "target_volatility_5d"]].tail(12))
    # the final rows will have NaN values for the target variable since we cannot compute a 5-day forward volatility for them
    print()
    print("Missing target values:")
    print(df["target_volatility_5d"].isnull().sum())
    
    df.to_csv(OUTPUT_PATH, index=False)
    
if __name__ == "__main__":
    main()
