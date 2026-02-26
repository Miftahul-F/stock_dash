import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
st.title("🧠 Institutional Swing System v3 – Fund Manager Grade")

# =============================
# INPUT
# =============================
st.sidebar.header("Scanner Settings")

watchlist_input = st.sidebar.text_area(
    "Watchlist (.JK dipisah koma)",
    value="BBRI.JK,BMRI.JK,TLKM.JK,ASII.JK,ADRO.JK,MDKA.JK"
)

tickers = [x.strip().upper() for x in watchlist_input.split(",")]

period = "6mo"
interval = "1d"

# =============================
# SAFE DATA LOADER
# =============================

def get_data(ticker):
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if df is None or df.empty:
        return None

    # FIX MultiIndex (critical for Streamlit Cloud)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Force numeric 1D series
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                pd.Series(df[col].squeeze(), index=df.index),
                errors="coerce"
            )

    df.dropna(subset=["Close"], inplace=True)

    if len(df) < 60:
        return None

    return df


# =============================
# INDICATORS
# =============================

def add_indicators(df):

    close = pd.Series(df["Close"].squeeze(), index=df.index)
    high  = pd.Series(df["High"].squeeze(), index=df.index)
    low   = pd.Series(df["Low"].squeeze(), index=df.index)
    vol   = pd.Series(df["Volume"].squeeze(), index=df.index)

    df["EMA20"] = ta.trend.EMAIndicator(close, 20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(close, 50).ema_indicator()
    df["RSI"] = ta.momentum.RSIIndicator(close, 14).rsi()

    macd = ta.trend.MACD(close)
    df["MACD_hist"] = macd.macd_diff()

    df["ATR"] = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range()
    df["VolMA20"] = vol.rolling(20).mean()

    df["Return20"] = close.pct_change(20)

    return df


# =============================
# MARKET REGIME (IHSG)
# =============================

ihsg = get_data("^JKSE")
if ihsg is None:
    st.error("Gagal ambil data IHSG")
    st.stop()

ihsg = add_indicators(ihsg)
ihsg_last = ihsg.iloc[-1]

market_bull = (
    ihsg_last["Close"] > ihsg_last["EMA50"] and
    ihsg_last["EMA20"] > ihsg_last["EMA50"]
)

if market_bull:
    st.success("Market Regime: BULLISH (IHSG sehat)")
else:
    st.warning("Market Regime: Tidak ideal (IHSG lemah/sideways)")


# =============================
# SCANNER
# =============================

results = []

with st.spinner("Scanning institutional candidates..."):
    for ticker in tickers:

        df = get_data(ticker)
        if df is None:
            continue

        df = add_indicators(df)

        row = df.iloc[-1]
        prev = df.iloc[-2]

        score = 0

        # Trend Alignment
        if row["EMA20"] > row["EMA50"]:
            score += 20

        # Relative Strength
        rs = row["Return20"] - ihsg_last["Return20"]
        if rs > 0:
            score += 15

        # Momentum
        if 45 <= row["RSI"] <= 60 and row["MACD_hist"] > prev["MACD_hist"]:
            score += 15

        # Volume Accumulation
        if row["Volume"] > 1.3 * row["VolMA20"]:
            score += 15

        # Structure (Higher Low)
        if row["Low"] > df["Low"].rolling(10).min().iloc[-2]:
            score += 10

        # Risk Model
        entry = row["Close"]
        sl = entry - row["ATR"]
        tp = entry + row["ATR"] * 2.2

        rr = (tp - entry) / (entry - sl) if (entry - sl) > 0 else 0

        if rr >= 1.8:
            score += 10

        # Market Regime Bonus
        if market_bull:
            score += 15

        if score >= 65:
            results.append({
                "Ticker": ticker,
                "Score": score,
                "Close": round(entry, 2),
                "RR": round(rr, 2),
                "RS_vs_IHSG": round(rs, 4),
                "Entry": round(entry, 2),
                "TP": round(tp, 2),
                "SL": round(sl, 2)
            })


# =============================
# OUTPUT
# =============================

if not results:
    st.warning("Tidak ada kandidat institutional valid hari ini.")
    st.stop()

df_result = pd.DataFrame(results).sort_values("Score", ascending=False)

st.subheader("📊 Institutional Swing Ranking")
selected = st.selectbox("Pilih Saham", df_result["Ticker"])
st.dataframe(df_result, use_container_width=True)


# =============================
# DETAIL CHART
# =============================

df = get_data(selected)
df = add_indicators(df)

row = df.iloc[-1]

entry = row["Close"]
sl = entry - row["ATR"]
tp = entry + row["ATR"] * 2.2

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.7, 0.3]
)

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"]
), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50"), row=1, col=1)

fig.add_hline(y=entry, line_color="blue", annotation_text="ENTRY")
fig.add_hline(y=tp, line_color="green", annotation_text="TP")
fig.add_hline(y=sl, line_color="red", annotation_text="SL")

fig.add_trace(go.Bar(x=df.index, y=df["Volume"]), row=2, col=1)

fig.update_layout(template="plotly_dark", height=850)

st.plotly_chart(fig, use_container_width=True)
