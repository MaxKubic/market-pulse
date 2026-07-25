# Market Pulse

Analytický nástroj propojující finanční sentiment (zprávy, Reddit) s pohyby cen akcií. Cílem projektu je otestovat, zda náhlé výkyvy sentimentu mají vztah k pozdějším cenovým pohybům, formou backtestu — nikoliv predikce budoucích cen.

## Cíl projektu

- Stažení a zpracování finančních titulků (NewsAPI) a diskuzí (historický Reddit dataset + živý RSS feed)
- Sentiment analýza pomocí FinBERT
- Spojení sentimentu s cenovými daty a volatilitou
- Backtesting jednoduché sentiment-based strategie proti benchmarku (buy-and-hold)
- Statistické vyhodnocení výsledků (Sharpe ratio, max drawdown, statistická významnost)
- Automatizovaný noční přepočet (GitHub Actions) a interaktivní dashboard (Streamlit)

## Struktura projektu

```
market-pulse/
├── data/
│   ├── raw/            # Nezpracovaná stažená data (negitovaná)
│   └── processed/       # Vyčištěná a spojená data
├── notebooks/            # Číslované notebooky (01–04) pro jednotlivé fáze analýzy
├── src/                   # Znovupoužitelné Python moduly (data loading, sentiment, backtest)
├── visuals/               # Exportované grafy a vizualizace
├── requirements.txt
└── .env.example           # Šablona pro API klíče (skutečný .env se negituje)
```

## Data

- **Ceny akcií:** yfinance (bez nutnosti API klíče)
- **Zprávy:** NewsAPI
- **Reddit sentiment (historický):** Kaggle dataset (r/wallstreetbets historické posty)
- **Reddit sentiment (živý):** RSS feed sledovaných subredditů

## Poznámka k metodologii

Backtest je navržen s důrazem na eliminaci look-ahead bias (sentiment je vždy časově zarovnán tak, aby předcházel signálu, ne naopak), zahrnuje transakční náklady a je porovnáván s benchmarkem. Výsledky jsou prezentovány včetně limitací, ne jako záruka budoucí výkonnosti.
