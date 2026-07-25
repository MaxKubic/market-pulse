"""
Stažení historických cen akcií.

Ukládá jeden CSV soubor na ticker do data/raw/prices/.
Časový rozsah je omezen na dostupnost Reddit sentiment dat (viz README),
takže defaultně stahujeme 2020-01-01 až dnes -- pokryje to i meme-stock
období, které je pro sentiment analýzu nejzajímavější.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

TICKERS = [
    "AAPL",
    "TSLA",
    "NVDA",
    "MSFT",
    "AMZN",
    "GME",
    "AMC",
    "PLTR",
    "DIS",
]

START_DATE = "2020-01-01"
END_DATE = None

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "prices"

def fetch_and_save(ticker: str) -> None:
    print(f"Stahuji {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

    if df.empty:
        print(f"  VAROVANI: pro {ticker} se nestahla zadna data.")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df["Ticker"] = ticker

    output_path = OUTPUT_DIR / f"{ticker}.csv"
    df.to_csv(output_path, index=False)
    print(f"  Ulozeno: {output_path} ({len(df)} radku)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        fetch_and_save(ticker)

    print("\nHotovo.")


if __name__ == "__main__":
    main()