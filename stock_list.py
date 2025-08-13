# streamlit_multi_tf.py
# Streamlit app: Multi-timeframe stock analyzer (1h, 4h, daily, weekly)
# Features:
# - Input ticker (global / .JK)
# - Pilih timeframe: 1h / 4h / Daily / Weekly
# - Input avg buy & lot
# - Hitung indikator MA9, RSI(14), MACD, ATR(14), Volume MA20
# - Auto S/R (pivot + swing high/low)
# - Entry (Buy on Weakness), TP, SL otomatis
# - Rekomendasi BUY / HOLD / SELL
# - Interactive Plotly charts

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="Multi-Timeframe Stock Analyzer", layout="wide")
st.title("📊 Multi-Timeframe Stock Analyzer (1h / 4h / Daily / Weekly)")

# -------------------------
# Sidebar / Inputs
# -------------------------
with st.sidebar:
    st.markdown("## Input & Pengaturan")
    ticker = st.text_input("Ticker (contoh: IKAN.JK atau AAPL)", value="IKAN.JK").upper().strip()
    timeframe = st.selectbox("Timeframe", options=["1h", "4h", "Daily", "Weekly"], index=2)
    # default periods
    if timeframe in ["1h", "4h"]:
        period_default = "60d"
    elif timeframe == "Daily":
        period_default = "1y"
    else:
        period_default = "5y"
    period = st.text_input("Period (yfinance, misal: 60d / 1y / 5y)", value=period_default)
    # portfolio inputs
    st.markdown("---")
    st.markdown("### Data Pembelian (opsional)")
    avg_buy = st.number_input("Avg Buy (Rp per lembar)", min_value=0.0, value=0.0, step=0.01)
    lots = st.number_input("Jumlah lot (1 lot = 100 lembar)", min_value=0, value=0, step=1)
    st.markdown("---")
    st.markdown("Tips:\n- Untuk BEI gunakan suffix `.JK` (mis. IKAN.JK)\n- Interval 4h diambil dari resample 60m -> 4H")

# -------------------------
# Helper functions
# -------------------------
def download_ohlcv(ticker, period, interval):
    """Download OHLCV and ensure DataFrame with Open,High,Low,Close,Volume columns (1D)."""
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    except Exception as e:
        st.error(f"Gagal download: {e}")
        return None
    if df is None or df.empty:
        return None

    # handle multiindex
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=-1)
        except Exception:
            # pick first levels
            df.columns = [c[0] for c in df.columns]
    # keep columns
    cols = [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
    df = df[cols].copy()

    # ensure numeric 1D Series
    for c in df.columns:
        df[c] = pd.to_numeric(pd.Series(df[c].squeeze(), index=df.index), errors='coerce')
    df.dropna(subset=['Close'], inplace=True)
    return df

def resample_4h(df_60m):
    """Resample 60m df to 4H (aggregation)."""
    df = df_60m.copy()
    # ensure timezone-naive and sorted
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df_4h = df.resample('4H').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'})
    df_4h.dropna(subset=['Close'], inplace=True)
    return df_4h

def compute_indicators(df):
    close = pd.Series(df['Close'].squeeze(), index=df.index)
    high  = pd.Series(df['High'].squeeze(), index=df.index) if 'High' in df else close
    low   = pd.Series(df['Low'].squeeze(), index=df.index)  if 'Low' in df else close
    vol   = pd.Series(df['Volume'].squeeze(), index=df.index) if 'Volume' in df else None

    # ensure numeric
    close = pd.to_numeric(close, errors='coerce')
    high = pd.to_numeric(high, errors='coerce')
    low = pd.to_numeric(low, errors='coerce')

    out = df.copy()
    out['MA9'] = close.rolling(9).mean()
    out['RSI14'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd_obj = ta.trend.MACD(close=close)
    out['MACD'] = macd_obj.macd()
    out['MACD_signal'] = macd_obj.macd_signal()
    # ATR
    try:
        atr_obj = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
        out['ATR14'] = atr_obj.average_true_range()
    except Exception:
        out['ATR14'] = np.nan
    # volume MA20
    if vol is not None:
        out['VolMA20'] = vol.rolling(20).mean()
    return out

def compute_sr_levels_lastbar(df):
    h=float(df['High'].iloc[-1]); l=float(df['Low'].iloc[-1]); c=float(df['Close'].iloc[-1])
    pivot=(h+l+c)/3.0
    r1=(2*pivot)-l; s1=(2*pivot)-h
    r2=pivot+(h-l); s2=pivot-(h-l)
    return pivot, r1, r2, s1, s2

def compute_entry_tp_sl(df, swing_window=10):
    c=float(df['Close'].iloc[-1])
    pivot,r1,r2,s1,s2 = compute_sr_levels_lastbar(df)
    swing_res = float(df['High'].tail(swing_window).max())
    swing_sup = float(df['Low'].tail(swing_window).min())
    atr = float(df['ATR14'].iloc[-1]) if 'ATR14' in df and not pd.isna(df['ATR14'].iloc[-1]) else c*0.02

    # entry candidates <= c
    candidates = [lvl for lvl in [df['MA9'].iloc[-1], pivot, s1, swing_sup] if (not pd.isna(lvl)) and (lvl <= c)]
    entry = max(candidates) if candidates else c*0.99
    # TP: nearest resistance >= c
    res_level = [lvl for lvl in [r1,r2,swing_res] if lvl >= c]
    tp = min(res_level) if res_level else c + 1.5*atr
    # SL: nearest support <= c
    sup_level = [lvl for lvl in [s1,s2,swing_sup] if lvl <= c]
    sl = max(sup_level) if sup_level else c - 1.0*atr

    # ensure decent RR
    risk = c - sl
    reward = tp - c
    if risk > 0 and (reward / risk) < 1.0:
        tp = c + max(risk * 1.2, 1.2*atr)
    return float(entry), float(tp), float(sl), (pivot,r1,r2,s1,s2,swing_res,swing_sup)

def simple_rekom(df_row):
    try:
        c = float(df_row['Close'])
        ma9 = float(df_row['MA9'])
        rsi = float(df_row['RSI14'])
        macd = float(df_row['MACD'])
        sig = float(df_row['MACD_signal'])
    except Exception:
        return "WAIT"
    if c > ma9 and rsi < 70 and macd > sig:
        return "BUY"
    if c < ma9 and rsi > 50 and macd < sig:
        return "SELL"
    return "HOLD"

# -------------------------
# Fetch data according to timeframe
# -------------------------
interval_map = {
    "1h": "60m",
    "4h": "60m",  # we'll resample
    "Daily": "1d",
    "Weekly": "1wk"
}

interval = interval_map.get(timeframe, "1d")
df_raw = download_ohlcv(ticker, period=period, interval=interval)
if df_raw is None:
    st.error("Tidak ada data. Periksa ticker atau koneksi internet.")
    st.stop()

# resample for 4h if requested
if timeframe == "4h":
    df = resample_4h(df_raw)
else:
    df = df_raw.copy()

# ensure enough bars
if df.shape[0] < 20:
    st.warning("Data terlalu sedikit untuk analisa (butuh minimal ~20 bar). Coba periode lebih panjang.")
    # continue to show what we have

# compute indicators
df_i = compute_indicators(df)

# compute levels & last values
last = df_i.iloc[-1]
last_close = float(last['Close'])
last_ma9 = float(last['MA9']) if not pd.isna(last['MA9']) else np.nan
last_rsi = float(last['RSI14']) if not pd.isna(last['RSI14']) else np.nan
last_macd = float(last['MACD']) if not pd.isna(last['MACD']) else np.nan
last_sig = float(last['MACD_signal']) if not pd.isna(last['MACD_signal']) else np.nan
last_atr = float(last['ATR14']) if 'ATR14' in last and not pd.isna(last['ATR14']) else np.nan
last_vol = int(last['Volume']) if 'Volume' in last else 0
last_volma = float(last['VolMA20']) if 'VolMA20' in last and not pd.isna(last['VolMA20']) else np.nan

# compute entry/tp/sl
entry, tp, sl, levels = compute_entry_tp_sl(df_i, swing_window=10)
pivot, r1, r2, s1, s2, swing_res, swing_sup = levels

# rekomendasi
rekom = simple_rekom(last)

# Portfolio calculations
shares = int(lots * 100)
if avg_buy > 0 and shares > 0:
    modal = avg_buy * shares
    nilai_now = last_close * shares
    pnl = nilai_now - modal
    pnl_pct = (pnl / modal) * 100 if modal != 0 else 0.0
else:
    modal = nilai_now = pnl = pnl_pct = 0

# -------------------------
# Top summary
# -------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Ticker", ticker)
col2.metric("Harga Sekarang", f"{last_close:,.2f}")
col3.metric("Rekomendasi", rekom)
col4.metric("Timeframe", timeframe)

st.markdown(f"**Entry (Buy on Weakness)**: {entry:.2f}  •  **TP**: {tp:.2f}  •  **SL**: {sl:.2f}")
st.markdown(f"**Pivot | R1 | R2**: {pivot:.2f} | {r1:.2f} | {r2:.2f}")
st.markdown(f"**Swing High(10d)**: {swing_res:.2f}  •  **Swing Low(10d)**: {swing_sup:.2f}")
if not np.isnan(last_atr):
    st.markdown(f"**ATR(14)**: {last_atr:.2f}  •  Volume (last): {last_vol:,}  •  VolMA20: {int(last_volma) if not np.isnan(last_volma) else '-'}")

if shares > 0:
    st.subheader("💼 Posisi Portofolio")
    st.write(f"- Qty: {shares} lembar  •  Modal: {modal:,.0f}  •  Nilai sekarang: {nilai_now:,.0f}")
    st.write(f"- P/L: {pnl:,.0f}  •  P/L %: {pnl_pct:.2f}%")
    if pnl_pct >= 10:
        st.success("💰 Profit > 10% — pertimbangkan ambil sebagian.")
    elif pnl_pct <= -5:
        st.warning("📉 Rugi > 5% — perhatikan support/sl.")

# -------------------------
# Alasan indikator singkat
# -------------------------
st.markdown("### 📌 Alasan (indikator)")
reasons = []
reasons.append(f"MA9: {last_ma9:.2f} → harga {'di atas' if last_close>last_ma9 else 'di bawah'} MA9")
if last_rsi < 30:
    reasons.append(f"RSI(14): {last_rsi:.2f} → Oversold")
elif last_rsi > 70:
    reasons.append(f"RSI(14): {last_rsi:.2f} → Overbought")
else:
    reasons.append(f"RSI(14): {last_rsi:.2f} → Netral")
reasons.append(f"MACD: {last_macd:.3f} vs Signal: {last_sig:.3f} → {'Bullish' if last_macd>last_sig else 'Bearish'}")
if not np.isnan(last_volma):
    vol_note = "tinggi" if last_vol > 1.2*last_volma else ("rendah" if last_vol < 0.8*last_volma else "normal")
    reasons.append(f"Volume: {last_vol:,} (MA20: {int(last_volma):,}) → {vol_note}")
for r in reasons:
    st.write(f"- {r}")

# -------------------------
# Charts: Candlestick + RSI + MACD + Volume
# -------------------------
st.markdown("### 📈 Chart Interaktif")
fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                    row_heights=[0.5, 0.12, 0.18, 0.18],
                    vertical_spacing=0.02,
                    subplot_titles=("Harga & MA9 + Level", "Volume", "RSI(14)", "MACD"))

# Price
fig.add_trace(go.Candlestick(x=df_i.index, open=df_i['Open'], high=df_i['High'],
                             low=df_i['Low'], close=df_i['Close'], name="Price"), row=1, col=1)
fig.add_trace(go.Scatter(x=df_i.index, y=df_i['MA9'], mode='lines', name='MA9', line=dict(color='orange')), row=1, col=1)
# lines
for y, txt, color in [(entry,"Entry","#3498db"), (tp,"TP","#2ecc71"), (sl,"SL","#e74c3c"),
                      (pivot,"Pivot","#7f8c8d"), (r1,"R1","#e67e22"), (r2,"R2","#d35400"),
                      (s1,"S1","#27ae60"), (s2,"S2","#16a085"),
                      (swing_res,"SwingHigh","#c0392b"), (swing_sup,"SwingLow","#2980b9")]:
    fig.add_hline(y=y, line_dash="dot", line_color=color, annotation_text=txt, annotation_position="top right", row=1, col=1)

# Volume
if 'Volume' in df_i.columns:
    fig.add_trace(go.Bar(x=df_i.index, y=df_i['Volume'], name='Volume', marker_color='lightblue'), row=2, col=1)
    if 'VolMA20' in df_i.columns:
        fig.add_trace(go.Scatter(x=df_i.index, y=df_i['VolMA20'], mode='lines', name='VolMA20', line=dict(color='orange')), row=2, col=1)

# RSI
fig.add_trace(go.Scatter(x=df_i.index, y=df_i['RSI14'], mode='lines', name='RSI', line=dict(color='yellow')), row=3, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

# MACD
fig.add_trace(go.Scatter(x=df_i.index, y=df_i['MACD'], mode='lines', name='MACD', line=dict(color='cyan')), row=4, col=1)
fig.add_trace(go.Scatter(x=df_i.index, y=df_i['MACD_signal'], mode='lines', name='Signal', line=dict(color='magenta')), row=4, col=1)
fig.add_hline(y=0, line_dash="dot", line_color="white", row=4, col=1)

fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=1000, showlegend=True)
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Footer / tips
# -------------------------
st.markdown("---")
st.markdown("**Catatan:**\n- Rekomendasi bersifat teknikal sederhana (MA9, RSI, MACD). Gunakan manajemen risiko.\n- Untuk intraday (1h/4h) data historis terbatas; perhatikan period yang kamu pilih.\n- Ingin notifikasi atau export otomatis? Beri tahu saya!")
