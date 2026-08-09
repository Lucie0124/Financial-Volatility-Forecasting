### Data Visualization for Market Risk Analysis ###

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

PATH_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PATH_ROOT / "data" / "processed" / "market_data_clean.csv"

df = pd.read_csv(INPUT_PATH, parse_dates=["date"])

plt.figure(figsize=(10, 6))

for column in ["vix", "treasury_10y", "treasury_2y"]:
    plt.plot(df["date"], df[column], label=column)
    
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Market Indicators Over Time")
plt.grid()
plt.legend()
plt.show()
