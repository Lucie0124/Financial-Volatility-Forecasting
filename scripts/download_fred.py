"""Download and merge the FRED series used by the project."""

from functools import reduce
from pathlib import Path

import pandas as pd


START_DATE = "2016-08-01"
END_DATE = "2026-07-31"

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

SERIES = {
    "SP500": "sp500",
    "VIXCLS": "vix",
    "DGS10": "treasury_10y",
    "DGS2": "treasury_2y",
    "BAMLH0A0HYM2": "high_yield_spread",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]  
RAW_DIRECTORY = PROJECT_ROOT / "data/raw"
PROCESSED_DIRECTORY = PROJECT_ROOT / "data/processed"


def download_series(series_id: str, column_name: str) -> pd.DataFrame:
    """Download one FRED series and return a standardised DataFrame."""
    url = (
        f"{FRED_CSV_URL}"
        f"?id={series_id}"
        f"&cosd={START_DATE}"
        f"&coed={END_DATE}"
    )

    dataframe = pd.read_csv(url, na_values=[".", ""])

    dataframe = dataframe.rename(columns={"observation_date": "date", series_id: column_name})

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="raise", # pandas raises an exception when it encounters an invalid date
    )
    dataframe[column_name] = pd.to_numeric(
        dataframe[column_name],
        errors="coerce", # pandas converts invalid values to NaN instead of raising an exception
    )

    dataframe = (
        dataframe[["date", column_name]]
        .drop_duplicates(subset="date")
        .sort_values("date")
        .reset_index(drop=True)
    ) # These parentheses allow one Python expression to be written across several lines

    return dataframe


def main() -> None:
    """Download individual CSV files and create a merged dataset."""
    
    # This creates the directories before attempting to save files
    # Without parents=True, Python can create raw/ only if data/ already exists
    # exist_ok=True : does not raise an error when the directory already exists
    RAW_DIRECTORY.mkdir(parents=True, exist_ok=True) 
    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True) 

    dataframes: list[pd.DataFrame] = [] # an empty list, will contain pandas DataFrames

    for series_id, column_name in SERIES.items():
        dataframe = download_series(series_id, column_name)

        raw_path = RAW_DIRECTORY / f"{series_id}.csv" # f-string : allowing Python variables to be inserted inside {}
        dataframe.to_csv(raw_path, index=False) # index=False : do not include row numbers to the CSV file 

        print(f"Saved {raw_path}")
        dataframes.append(dataframe)

    # Merge all the individual DataFrames into one DataFrame
    merged = reduce(
        lambda left, right: pd.merge( # lambda function : a small anonymous function 
            left,
            right,
            on="date",
            how="outer", # outer join : keeps all rows from both DataFrames, filling in NaN for missing values
            validate="one_to_one", # ensures that each date appears at most once in each DataFrame
        ),
        dataframes, # the list of DataFrames to merge
    )

    # sort the merged DataFrame by date and reset the index, dropping the old index
    merged = merged.sort_values("date").reset_index(drop=True) 
    
    output_path = PROCESSED_DIRECTORY / "market_data.csv"
    merged.to_csv(output_path, index=False)

    print()
    print(f"Rows: {len(merged):,}")
    print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")

    missing_values = merged.isna().sum()
    missing_percent = merged.isna().mean() * 100
    
    print()
    for column in merged.columns:
        print(f"{column}: {missing_values[column]}"
              f"({missing_percent[column]:.2f}%)")

if __name__ == "__main__":
    main()
    
    
