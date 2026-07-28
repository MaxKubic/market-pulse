# Market Pulse

Analytický nástroj propojující finanční sentiment (Reddit) s pohyby cen akcií. Cílem projektu je otestovat, zda náhlé výkyvy sentimentu mají vztah k pozdějším cenovým pohybům, formou backtestu — nikoliv predikce budoucích cen.

## 🔑 Klíčové zjištění

Systematický backtest (viz `notebooks/03_backtest.ipynb` a `notebooks/04_strategy_variants.ipynb`) **neprokázal, že by jednoduchá sentiment-based strategie měla statisticky významnou výhodu** nad náhodným časováním obchodů (permutační test, p-hodnota 0.13–0.38 v závislosti na konfiguraci parametrů).

Zajímavé vedlejší zjištění: strategie z velké části **propásla GME short squeeze** (leden 2021) — přesně tu epizodu extrémního sentimentu, na kterou má být citlivá. Možné vysvětlení: krátký (1denní) holding period nestačí zachytit pokračující vícedenní růst, nebo je definice signálu příliš konzervativní právě v okamžicích největší volatility.

Tenhle negativní/neprůkazný výsledek je stejně cenný jako pozitivní — ukazuje, že syrový Reddit sentiment sám o sobě není dostatečně silný prediktivní signál bez dalšího zpracování (např. delší holding period, kombinace s objemem/momentum, sofistikovanější NLP).

## Metodologické highlighty

- **Look-ahead bias:** sentiment zjištěný k datu T smí ovlivnit jen obchod na T+1 nebo později; víkendový/sváteční sentiment se přiřazuje nejbližšímu následujícímu obchodnímu dni (`pd.merge_asof`)
- **Volba sentiment modelu:** původně zvažován FinBERT, ale rychlý test odhalil, že špatně zvládá Reddit slang ("diamond hands", "to the moon" → mylně klasifikováno jako neutrální). Přepnuto na `cardiffnlp/twitter-roberta-base-sentiment-latest`, který zvládá neformální jazyk výrazně lépe (viz `src/compare_sentiment_models.py` a `notebooks/02_sentiment_analysis.ipynb`)
- **Oříznutí na skutečné pokrytí dat:** cenová data se stahují až do současnosti, ale historický Reddit dataset končí únorem 2021. Analýza je oříznutá na období se skutečným sentiment pokrytím — jinak by chronologický train/test split mohl umístit celé testovací období do měsíců bez jakéhokoliv signálu (což se při prvním pokusu skutečně stalo)
- **Transakční náklady, benchmark, rizikové metriky:** Sharpe ratio, max drawdown, win rate, srovnání s buy-and-hold benchmarkem
- **Permutační test + out-of-sample split:** oddělené ladění parametrů (trénovací data) od jediného ověření (testovací data) — žádné "zkoušej, dokud to nevyjde"

## Cíl projektu

- Stažení a zpracování cenových dat (yfinance) a Reddit diskuzí (historický Kaggle dataset)
- Sentiment analýza pomocí NLP modelu
- Spojení sentimentu s cenovými daty a volatilitou, s důrazem na look-ahead-bias-safe metodologii
- Backtesting sentiment-based strategie proti benchmarku (buy-and-hold)
- Statistické vyhodnocení výsledků (Sharpe ratio, max drawdown, permutační test, out-of-sample validace)
- Interaktivní Streamlit dashboard pro prozkoumání různých nastavení strategie

## Struktura projektu

```
market-pulse/
├── data/
│   ├── raw/              # Nezpracovaná stažená data (negitovaná)
│   └── processed/        # Vyčištěná a spojená data (negitovaná)
├── notebooks/
│   ├── 01_data_cleaning.ipynb        # Čištění cen + Reddit dat, tagování tickerů
│   ├── 02_sentiment_analysis.ipynb   # Sentiment scoring (Twitter-RoBERTa)
│   ├── 03_backtest.ipynb             # Spojení dat, look-ahead-bias-safe backtest
│   └── 04_strategy_variants.ipynb    # Parameter sweep (train) + jediné ověření (test)
├── src/
│   ├── fetch_prices.py               # Stažení cen přes yfinance
│   ├── inspect_reddit_data.py        # Validace pokrytí Reddit datasetu
│   └── compare_sentiment_models.py   # FinBERT vs. Twitter-RoBERTa srovnání
├── app.py                            # Streamlit dashboard
├── visuals/
├── requirements.txt
└── .env.example                      # Šablona pro API klíče (skutečný .env se negituje)
```

## Data

- **Ceny akcií:** yfinance (bez nutnosti API klíče), tickery: AAPL, TSLA, NVDA, MSFT, AMZN, GME, AMC, PLTR, DIS
- **Reddit sentiment:** historický Kaggle dataset (r/wallstreetbets posty, 2012–2021, oříznuto na období 2020-01 až 2021-02 pro analýzu)

Reddit API self-service registrace byla v době vzniku projektu uzavřená (viz [Reddit Responsible Builder Policy](https://support.reddithelp.com/hc/en-us/articles/42728983564564)), proto se pro historický backtest používá hotový Kaggle dataset místo živého API.

## Jak spustit

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

1. Doplň `.env` podle `.env.example` (NewsAPI klíč, pokud budeš rozšiřovat o news sentiment)
2. Stáhni Reddit dataset (Kaggle: `unanimad/reddit-rwallstreetbets` nebo podobný) do `data/raw/reddit/`
3. Spusť notebooky v pořadí `01` → `02` → `03` → `04` (Jupyter/VS Code)
4. Spusť dashboard:
```bash
streamlit run app.py
```

## Limitace

- Forward return počítá s obchodem přesně na Close, realisticky by šlo o Open T+1 (dodatečný slippage)
- Strategie jen nakupuje, nikdy neshortuje
- Krátké analyzované období (~13 měsíců), navíc netypické (COVID + meme-stock mania) — závěry se nemusí generalizovat
- Přežitý dataset (survivorship bias) — všechny sledované tickery existují dodnes
- Sentiment model není doladěný na finanční/trading slang, jen obecně na social media jazyk

## Možné rozšíření

- Živý zdroj Reddit dat (RSS feed) místo historického datasetu — otevřelo by to prostor pro GitHub Actions automatizaci (noční přepočet), která u čistě historických dat nedává smysl
- Long/short strategie místo jen long-only
- Walk-forward validace s více nezávislými testovacími okny místo jednoho train/test splitu
- Fine-tuning sentiment modelu přímo na finančním/trading slangu
