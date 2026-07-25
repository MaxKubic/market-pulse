"""
Rychla inspekce stazeneho Reddit datasetu z Kaggle.

Zjistuje:
- jake sloupce dataset ma
- jake je casove rozpeti dat
- kolik zminek maji jednotlive sledovane tickery (podle nazvu firmy i symbolu)
"""

import pandas as pd
from pathlib import Path

REDDIT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "reddit"

# Tickery + alternativni nazvy/zkratky
TICKER_ALIASES = {
    "AAPL": ["AAPL", "Apple"],
    "TSLA": ["TSLA", "Tesla"],
    "NVDA": ["NVDA", "Nvidia"],
    "MSFT": ["MSFT", "Microsoft"],
    "AMZN": ["AMZN", "Amazon"],
    "GME": ["GME", "Gamestop", "GameStop"],
    "AMC": ["AMC"],
    "PLTR": ["PLTR", "Palantir"],
    "DIS": ["DIS", "Disney"],
}

def find_csv() -> Path:
    csvs = list(REDDIT_DIR.glob("r_wallstreetbets_posts.csv"))
    if not csvs:
        raise FileNotFoundError(f"Zadny CSV soubor nenalezen v {REDDIT_DIR}")
    if len(csvs) > 1:
        print(f"Pozor: nalezeno vic CSV souboru, pouzivam prvni: {csvs[0].name}")
    return csvs[0]

def find_text_column(df: pd.DataFrame) -> str:
    candidates = ["title", "Title", "text", "body", "selftext"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Nenasel jsem sloupec s textem. Dostupne sloupce: {list(df.columns)}"
    )

def find_date_column(df: pd.DataFrame) -> str:
    candidates = ["timestamp", "created", "created_utc", "date", "Date"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    csv_path = find_csv()
    print(f"Nacitam: {csv_path.name}\n")

    df = pd.read_csv(csv_path, low_memory=False)

    print(f"Pocet radku: {len(df)}")
    print(f"Sloupce: {list(df.columns)}\n")

    date_col = find_date_column(df)
    if date_col:
        if date_col in ("created_utc", "created"):
            dates = pd.to_datetime(df[date_col], unit="s", errors="coerce", utc=True)
        else:
            dates = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        print(f"Sloupec s datem: '{date_col}'")
        print(f"Casove rozpeti: {dates.min()} az {dates.max()}\n")
    else:
        print("Nenasel jsem zjevny sloupec s datem -- zkontroluj sloupce rucne.\n")

    text_col = find_text_column(df)
    print(f"Pouzivam sloupec s textem: '{text_col}'\n")

    text_lower = df[text_col].astype(str).str.lower()

    print("Pocet zminek podle tickeru:")
    print("-" * 40)
    for ticker, aliases in TICKER_ALIASES.items():
        mask = pd.Series(False, index=df.index)
        for alias in aliases:
            escaped = alias.lower().replace("&", r"\&")
            pattern = r"\b" + escaped + r"\b"
            mask |= text_lower.str.contains(pattern, regex=True, na=False)
        print(f"  {ticker:6s} {mask.sum():>8d} zminek")

if __name__ == "__main__":
    main()