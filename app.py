"""
Market Pulse -- interaktivni dashboard

Spusteni: streamlit run app.py (z korene projektu)

Nacita zpracovana data z data/processed/ (musis mit predtim spustene
notebooky 01-03, jinak soubory neexistuji).
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Market Pulse", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRANSACTION_COST = 0.001


@st.cache_data
def load_data():
    merged_path = PROCESSED_DIR / "merged_price_sentiment.csv"
    if not merged_path.exists():
        return None
    df = pd.read_csv(merged_path, parse_dates=["Date"])
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def compute_metrics(returns: pd.Series, active_mask=None) -> dict:
    if len(returns) == 0 or returns.std() == 0:
        return {"Celkovy vynos": "N/A", "Sharpe ratio": "N/A", "Max drawdown": "N/A", "Win rate": "N/A"}

    ann_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    if active_mask is not None and active_mask.sum() > 0:
        win_rate = (returns[active_mask] > 0).mean()
    else:
        win_rate = (returns > 0).mean()

    return {
        "Celkovy vynos": f"{(equity.iloc[-1] - 1) * 100:.1f} %",
        "Sharpe ratio": f"{sharpe:.2f}",
        "Max drawdown": f"{max_dd * 100:.1f} %",
        "Win rate": f"{win_rate * 100:.1f} %",
    }


st.title("📊 Market Pulse")
st.caption("Sentiment-driven analyza akciovych trhu -- FinBERT/Twitter-RoBERTa + backtesting")

df = load_data()

if df is None:
    st.error(
        "Zpracovana data nenalezena. Nejdriv spust notebooky 01_data_cleaning, "
        "02_sentiment_analysis a 03_backtest, at se vygeneruje "
        "data/processed/merged_price_sentiment.csv."
    )
    st.stop()

data_start = df["Date"].min().date()
data_end = df["Date"].max().date()
st.info(
    f"📅 Data pokryvaji obdobi **{data_start} az {data_end}** -- ohranicene dostupnosti "
    "historickeho Reddit datasetu, ne cenovych dat (viz README pro vysvetleni)."
)

# ---------------------------------------------------------------------------
# Sidebar -- ovladani parametru strategie
# ---------------------------------------------------------------------------
st.sidebar.header("Parametry strategie")

sentiment_threshold = st.sidebar.slider("Sentiment z-score prah", 0.5, 3.0, 1.5, 0.1)
volume_threshold = st.sidebar.slider("Volume z-score prah", 0.0, 2.0, 1.0, 0.1)
holding_period = st.sidebar.selectbox("Holding period (dny)", [1, 3, 5], index=0)

st.sidebar.markdown("---")
tickers = sorted(df["Ticker"].unique())
selected_ticker = st.sidebar.selectbox("Ticker pro detailni graf", tickers)

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Toto je edukativni/portfolio nastroj, ne investicni doporuceni. "
    "Backtest v tomto projektu neprokazal statisticky vyznamnou prevahu "
    "nad nahodnym casovanim obchodu (viz notebook 03/04)."
)

# ---------------------------------------------------------------------------
# Vypocet strategie s aktualnimi parametry
# ---------------------------------------------------------------------------
work = df.copy()
work[f"Forward_Return_{holding_period}d"] = (
    work.groupby("Ticker")["Close"].shift(-holding_period) / work["Close"] - 1
)
return_col = f"Forward_Return_{holding_period}d"

work["signal"] = (
    (work["sentiment_zscore"] > sentiment_threshold)
    & (work["post_count_zscore"] > volume_threshold)
).astype(int)

all_dates = pd.Series(sorted(work["Date"].unique()))

daily_strategy = work[work["signal"] == 1].groupby("Date")[return_col].mean()
strategy_returns = all_dates.map(daily_strategy).fillna(0.0)
strategy_returns.index = all_dates

trade_days = all_dates.isin(daily_strategy.index)
strategy_returns_net = strategy_returns - (trade_days.values * TRANSACTION_COST)

benchmark_returns = work.groupby("Date")[return_col].mean().reindex(all_dates).fillna(0.0)
benchmark_returns.index = all_dates

strategy_equity = (1 + strategy_returns_net).cumprod()
benchmark_equity = (1 + benchmark_returns).cumprod()

n_trades = int(trade_days.sum())

# ---------------------------------------------------------------------------
# Hlavni panel -- equity krivka a metriky
# ---------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Strategie vs. benchmark")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(strategy_equity.index, strategy_equity.values, label="Sentiment strategie", linewidth=1.5)
    ax.plot(benchmark_equity.index, benchmark_equity.values, label="Benchmark (buy & hold)", linewidth=1.5, alpha=0.7)
    ax.set_ylabel("Kumulativni hodnota (start = 1.0)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Metriky")
    st.metric("Pocet aktivnich obchodnich dni", n_trades)

    metrics_strategy = compute_metrics(strategy_returns_net, active_mask=trade_days.values)
    metrics_benchmark = compute_metrics(benchmark_returns)

    metrics_df = pd.DataFrame(
        {"Strategie": metrics_strategy, "Benchmark": metrics_benchmark}
    )
    st.dataframe(metrics_df, use_container_width=True)

    if n_trades < 20:
        st.warning(
            f"Jen {n_trades} aktivnich obchodnich dni pri techto parametrech -- "
            "vysledky nejsou statisticky spolehlive (male vzorky davaji "
            "nahodne extremni vysledky)."
        )

# ---------------------------------------------------------------------------
# Detail vybraneho tickeru
# ---------------------------------------------------------------------------
st.subheader(f"Detail: {selected_ticker}")

ticker_data = work[work["Ticker"] == selected_ticker].copy()

fig2, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(ticker_data["Date"], ticker_data["Close"], color="steelblue", linewidth=1.2, label="Cena (Close)")
ax1.set_ylabel("Cena ($)", color="steelblue")

signal_dates = ticker_data[ticker_data["signal"] == 1]
ax1.scatter(
    signal_dates["Date"], signal_dates["Close"],
    color="crimson", zorder=5, s=40, label="Aktivni signal"
)

ax2 = ax1.twinx()
ax2.plot(ticker_data["Date"], ticker_data["sentiment_zscore"], color="orange", alpha=0.4, linewidth=0.8, label="Sentiment z-score")
ax2.axhline(sentiment_threshold, color="orange", linestyle="--", linewidth=0.8, alpha=0.6)
ax2.set_ylabel("Sentiment z-score", color="orange")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

st.pyplot(fig2)
plt.close(fig2)

st.caption(
    "Cervene body oznacuji dny, kdy strategie s aktualnim nastavenim parametru "
    "vygenerovala nakupni signal. Oranzova cara ukazuje sentiment z-score a "
    "prerusovana cara aktualni prah."
)

# ---------------------------------------------------------------------------
# Poctive shrnuti na konci
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander("📋 Metodologie a limitace (klikni pro rozbaleni)"):
    st.markdown("""
    **Co tento backtest dela:** simuluje jednoduchou strategii, ktera nakupuje
    akcie den po neobvykle pozitivnim a objemnem vykyvu Reddit sentimentu,
    a porovnava ji s pasivnim drzenim portfolia (buy & hold).

    **Klicove metodologicke zasady:**
    - Signal zjisteny k datu T ovlivnuje jen obchod na den T+1 nebo pozdeji
      (zadny look-ahead bias)
    - Vikendovy/svatecni sentiment se pripisuje nejblizsimu dalsimu
      obchodnimu dni
    - Kazdy obchod je zatizeny transakcnimi naklady (0.1 %)

    **Hlavni zjisteni projektu:** systematicky backtest (viz notebook 03/04)
    neprokazal, ze by tato jednoducha sentiment strategie mela statisticky
    vyznamnou prevahu nad nahodnym casovanim obchodu (permutacni test,
    p-hodnota v radu 0.1-0.4 v zavislosti na konfiguraci). Zajimave je, ze
    strategie casto propasla nejextremnejsi obdobi (GME short squeeze,
    leden 2021) -- prave to, na co ma byt citliva.

    **Data pokryvaji jen ~13 mesicu** (leden 2020 -- unor 2021), ohranicene
    dostupnosti historickeho Reddit datasetu. Vysledky se nemusi generalizovat
    na jina obdobi nebo trzni podminky.
    """)