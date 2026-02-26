import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pytz
import math
import re
from datetime import datetime
from PyPDF2 import PdfReader

st.set_page_config(layout="wide")
st.title("📊 Institutional Quant System – ISSI Auto Update")

WIB = pytz.timezone("Asia/Jakarta")

# =============================
# SETTINGS
# =============================
modal = st.sidebar.number_input("Modal (Rp)", value=100_000_000)
risk_percent = st.sidebar.slider("Risk per Trade (%)", 0.5, 3.0, 1.0)

# =============================
# EXTRACT ISSI FROM PDF
# =============================
@st.cache_data(ttl=3600)
def extract_issi_from_pdf(pdf_path):

    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    # Ambil kode saham 4 huruf kapital
    codes = set(re.findall(r"\b[A-Z]{4}\b", text))

    # Filter out non-ticker noise (contoh umum)
    blacklist = ["BEI","ISSI","JII","IDX","MBX","DBX","ABX"]
    codes = [c for c in codes if c not in blacklist]

    return sorted(list(codes))

try:
    ISSI_CODES = extract_issi_from_pdf("issi_latest.pdf")
    ISSI_UNIVERSE = [x + ".JK" for x in ISSI_CODES]
    st.success(f"ISSI Universe Loaded: {len(ISSI_UNIVERSE)} saham")
except:
    st.error("File issi_latest.pdf tidak ditemukan.")
    st.stop()

# =============================
# SAFE DATA LOADER
# =============================
@st.cache_data(ttl=3600)
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

    df.dropna(inplace=True)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(WIB)

    return df

# =============================
# STEP 1 – LIQUIDITY RANKING
# =============================
st.subheader("🔎 Generating Top150 Liquid ISSI...")

liquidity_data = []

for ticker in ISSI_UNIVERSE:

    df = get_data(ticker, "1d", "3mo")
    if df is None or len(df) < 40:
        continue

    avg_value = (df["Close"] * df["Volume"]).mean()

    liquidity_data.append({
        "Ticker": ticker,
        "AvgValue": avg_value
    })

df_liq = pd.DataFrame(liquidity_data)

df_liq = df_liq.sort_values("AvgValue", ascending=False)

top150 = df_liq.head(150)["Ticker"].tolist()

st.success(f"Top150 ISSI selected ({len(top150)} saham)")

# =============================
# STEP 2 – QUANT RANKING
# =============================
ranking = []

ihsg = get_data("^JKSE")
ihsg_ret = ihsg["Close"].pct_change(20).iloc[-1]

for ticker in top150:

    df = get_data(ticker, "1d", "6mo")
    if df is None or len(df) < 60:
        continue

    close = df["Close"]

    df["EMA20"] = ta.trend.EMAIndicator(close,20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(close,50).ema_indicator()
    df["RSI"] = ta.momentum.RSIIndicator(close,14).rsi()
    df["Return20"] = close.pct_change(20)

    row = df.iloc[-1]

    score = 0

    if row["EMA20"] > row["EMA50"]:
        score += 30

    if 45 <= row["RSI"] <= 65:
        score += 20

    if row["Return20"] > ihsg_ret:
        score += 25

    if row["Volume"] > df["Volume"].rolling(20).mean().iloc[-1]:
        score += 25

    ranking.append({
        "Ticker": ticker,
        "Score": score
    })

df_rank = pd.DataFrame(ranking).sort_values("Score", ascending=False)

st.subheader("📈 Top Quant Ranking")
st.dataframe(df_rank.head(10), use_container_width=True)

# =============================
# STEP 3 – EXECUTION MODEL
# =============================
st.subheader("🎯 Execution Model (Top 5)")

top_exec = df_rank.head(5)["Ticker"].tolist()

for ticker in top_exec:

    df = get_data(ticker, "1h", "30d")
    if df is None or len(df) < 20:
        continue

    df["EMA20"] = ta.trend.EMAIndicator(df["Close"],20).ema_indicator()
    df["ATR"] = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"], 14
    ).average_true_range()

    row = df.iloc[-1]

    if row["Close"] > row["EMA20"]:

        buffer = 0.001 * row["Close"]
        entry = row["High"] + buffer
        sl = entry - row["ATR"]
        risk = entry - sl
        tp = entry + risk * 2.2

        risk_amount = modal * (risk_percent / 100)
        size = math.floor((risk_amount / risk) / 100) * 100

        st.write(f"### {ticker}")
        st.write(f"Entry Stop : Rp {entry:,.0f}")
        st.write(f"Stop Loss  : Rp {sl:,.0f}")
        st.write(f"Take Profit: Rp {tp:,.0f}")
        st.write(f"Position Size: {size:,} saham ({size//100} lot)")
        st.write("---")
