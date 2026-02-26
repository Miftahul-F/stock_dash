import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from datetime import datetime
import pytz
import math

st.set_page_config(layout="wide")
st.title("🧠 Institutional Swing System v5 – Execution Ready")

# =========================
# SETTINGS
# =========================
st.sidebar.header("Scanner Settings")

score_threshold = st.sidebar.slider("Minimum Daily Score", 50, 90, 65)

modal = st.sidebar.number_input("Modal (Rp)", value=100_000_000)
risk_percent = st.sidebar.slider("Risk per Trade (%)", 0.5, 3.0, 1.0)

use_full_list = st.sidebar.checkbox("Gunakan List Saham Likuid IHSG", True)

manual_list = st.sidebar.text_area(
    "Manual Watchlist",
    value="BBRI.JK,BMRI.JK,BBCA.JK,ASII.JK"
)

if use_full_list:
    tickers = [
        "BBRI.JK","BMRI.JK","BBCA.JK","ASII.JK",
        "TLKM.JK","ADRO.JK","MDKA.JK","INDF.JK",
        "CPIN.JK","UNTR.JK"
    ]
else:
    tickers = [x.strip().upper() for x in manual_list.split(",")]

WIB = pytz.timezone("Asia/Jakarta")

def now_wib():
    return datetime.now(WIB)

# =========================
# SAFE DATA LOADER (WIB)
# =========================
def get_data(ticker, interval="1d", period="6mo"):
    df = yf.download(
        ticker,
        interval=interval,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    for col in ["Open","High","Low","Close","Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                pd.Series(df[col].squeeze(), index=df.index),
                errors="coerce"
            )

    df.dropna(subset=["Close"], inplace=True)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df.index = df.index.tz_convert(WIB)

    return df

# =========================
# INDICATORS
# =========================
def add_daily_indicators(df):
    close = df["Close"]
    df["EMA20"] = ta.trend.EMAIndicator(close,20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(close,50).ema_indicator()
    df["RSI"] = ta.momentum.RSIIndicator(close,14).rsi()
    df["Return20"] = close.pct_change(20)
    return df

def add_1h_indicators(df):
    close = df["Close"]
    df["EMA20"] = ta.trend.EMAIndicator(close,20).ema_indicator()
    macd = ta.trend.MACD(close)
    df["MACD_hist"] = macd.macd_diff()
    df["VolMA20"] = df["Volume"].rolling(20).mean()
    df["ATR"] = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"], 14
    ).average_true_range()
    return df

# =========================
# MARKET REGIME
# =========================
ihsg = get_data("^JKSE")
ihsg = add_daily_indicators(ihsg)
ihsg_last = ihsg.iloc[-1]

market_bull = ihsg_last["Close"] > ihsg_last["EMA50"]

st.markdown(f"### Market Status: {'🟢 BULLISH' if market_bull else '🟠 NOT IDEAL'}")
st.markdown(f"IHSG Last Update (WIB): {ihsg.index[-1]}")

# =========================
# DAILY FILTER
# =========================
results = []

with st.spinner("Scanning Daily Filter..."):
    for ticker in tickers:

        df = get_data(ticker)
        if df is None or len(df) < 60:
            continue

        df = add_daily_indicators(df)
        row = df.iloc[-1]

        score = 0

        if row["Close"] > row["EMA50"]:
            score += 25

        if row["EMA20"] > row["EMA50"]:
            score += 20

        if 45 <= row["RSI"] <= 65:
            score += 15

        rs = row["Return20"] - ihsg_last["Return20"]
        if rs > 0:
            score += 15

        if market_bull:
            score += 15

        results.append({
            "Ticker": ticker,
            "Score": score
        })

df_rank = pd.DataFrame(results).sort_values("Score", ascending=False)
st.subheader("📊 Daily Ranking")
st.dataframe(df_rank, use_container_width=True)

df_valid = df_rank[df_rank["Score"] >= score_threshold]

if df_valid.empty:
    st.warning("Tidak ada kandidat sesuai threshold.")
    st.stop()

selected = st.selectbox("Pilih Saham untuk Entry 1H", df_valid["Ticker"])

# =========================
# 1H ENTRY MODEL
# =========================
df_1h = get_data(selected, "1h", "30d")
df_1h = add_1h_indicators(df_1h)

row = df_1h.iloc[-1]
prev = df_1h.iloc[-2]

entry_signal = (
    row["Close"] > row["EMA20"] and
    row["MACD_hist"] > prev["MACD_hist"] and
    row["Volume"] > row["VolMA20"]
)

st.markdown(f"### 1H Signal: {'🟢 VALID' if entry_signal else '🟠 WAIT'}")
st.markdown(f"Last 1H Update (WIB): {df_1h.index[-1]}")

# =========================
# DELAY CHECK
# =========================
delay_minutes = (now_wib() - df_1h.index[-1]).total_seconds() / 60

if delay_minutes > 20:
    st.error("🚨 WARNING: Data delay > 20 menit")
else:
    st.success("Data relatif fresh")

# =========================
# EXECUTION CALCULATION
# =========================
if entry_signal:

    buffer = 0.001 * row["Close"]  # 0.1% buffer
    entry_price = row["High"] + buffer

    atr = row["ATR"]
    sl = entry_price - atr

    risk_per_share = entry_price - sl
    risk_amount = modal * (risk_percent / 100)

    position_size = risk_amount / risk_per_share
    position_size = math.floor(position_size / 100) * 100  # round lot BEI

    tp = entry_price + (risk_per_share * 2.2)

    st.subheader("🎯 Execution Plan")
    st.write(f"Entry (Buy Stop): Rp {entry_price:,.0f}")
    st.write(f"Stop Loss: Rp {sl:,.0f}")
    st.write(f"Take Profit (2.2R): Rp {tp:,.0f}")
    st.write(f"Risk per Share: Rp {risk_per_share:,.0f}")
    st.write(f"Position Size: {position_size:,} saham ({position_size//100} lot)")
    st.write(f"Risk Amount: Rp {risk_amount:,.0f}")

# =========================
# CHART
# =========================
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df_1h.index,
    open=df_1h["Open"],
    high=df_1h["High"],
    low=df_1h["Low"],
    close=df_1h["Close"]
))

fig.add_trace(go.Scatter(
    x=df_1h.index,
    y=df_1h["EMA20"],
    name="EMA20"
))

fig.update_xaxes(tickformat="%d-%m-%Y %H:%M", title="Waktu (WIB)")
fig.update_layout(template="plotly_dark", height=600)

st.plotly_chart(fig, use_container_width=True)