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
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 Institutional Semi-Auto Swing v8 (Top 3 Rotation)")

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
        if page.extract_text():
            text += page.extract_text()

    # Ambil kode 3-5 huruf kapital
    codes = set(re.findall(r"\b[A-Z]{3,5}\b", text))

    blacklist = [
        "BEI","ISSI","JII","IDX","MBX","DBX","ABX",
        "Lampiran","Evaluasi","Minor","Mayor"
    ]

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
# DATA LOADER
# =============================
@st.cache_data(ttl=1800)
def get_data(ticker, interval="1d", period="6mo"):
    df = yf.download(ticker, interval=interval, period=period,
                     auto_adjust=True, progress=False)

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
# STEP 1 – TOP150 LIQUID
# =============================
st.subheader("🔎 Generating Top150 Liquid ISSI...")

liquidity = []

for ticker in ISSI_UNIVERSE:
    df = get_data(ticker, "1d", "3mo")
    if df is None or len(df) < 40:
        continue

    avg_value = (df["Close"] * df["Volume"]).mean()

    liquidity.append({
        "Ticker": ticker,
        "AvgValue": avg_value
    })

df_liq = pd.DataFrame(liquidity)

if df_liq.empty:
    st.warning("Tidak ada data liquidity.")
    st.stop()

df_liq = df_liq.sort_values("AvgValue", ascending=False)
top150 = df_liq.head(150)["Ticker"].tolist()

st.success(f"Top150 ISSI selected ({len(top150)} saham)")

# =============================
# STEP 2 – DAILY QUANT RANKING
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

st.subheader("📈 Top 3 Rotation")
top3 = df_rank.head(3)
st.dataframe(top3, use_container_width=True)

# =============================
# STEP 3 – EXECUTION MODEL
# =============================
st.subheader("🎯 Execution Plan (Breakout 1H Only)")

for ticker in top3["Ticker"]:

    df = get_data(ticker, "1h", "30d")
    if df is None or len(df) < 20:
        continue

    df["EMA20"] = ta.trend.EMAIndicator(df["Close"],20).ema_indicator()
    df["ATR"] = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"], 14
    ).average_true_range()

    row = df.iloc[-1]
    last_time = df.index[-1]
    now_time = datetime.now(WIB)
    delay = (now_time - last_time).total_seconds() / 60

    # ENTRY BREAKOUT
    buffer = 0.001 * row["Close"]
    entry = row["High"] + buffer
    sl = entry - row["ATR"]
    risk = entry - sl
    tp = entry + risk * 2

    risk_amount = modal * (risk_percent / 100)
    raw_size = risk_amount / risk

    if raw_size < 100:
        size = 100
    else:
        size = math.floor(raw_size / 100) * 100

    st.markdown(f"### {ticker}")
    st.write(f"Entry Stop : Rp {entry:,.0f}")
    st.write(f"Stop Loss  : Rp {sl:,.0f}")
    st.write(f"Take Profit: Rp {tp:,.0f}")
    st.write(f"Position Size: {size:,} saham ({size//100} lot)")
    st.write(f"Data Delay: {round(delay,1)} menit")
    st.write("Max Hold: 3 hari")
    
    # =============================
    # CHART
    # =============================
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["EMA20"],
        name="EMA20"
    ))

    fig.add_hline(y=entry, line_dash="dot", annotation_text="Entry")
    fig.add_hline(y=sl, line_dash="dot", annotation_text="SL")
    fig.add_hline(y=tp, line_dash="dot", annotation_text="TP")

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Waktu (WIB)",
        yaxis_title="Harga"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.write("---")