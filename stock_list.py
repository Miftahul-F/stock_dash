# app_stock_screener.py
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ======================
# Util ambil data 1D
# ======================
def get_ohlcv(ticker: str, period="3mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    # Handle kemungkinan MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=-1)
        except Exception:
            cols0 = df.columns.get_level_values(0)
            keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in set(cols0)]
            df = df[keep]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
    keep_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    df = df[keep_cols].copy()
    # Pastikan kolom numeric & 1D
    for c in keep_cols:
        s = pd.Series(df[c].squeeze(), index=df.index)
        df[c] = pd.to_numeric(s, errors='coerce')
    df = df.dropna(subset=['Close'])
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = pd.Series(out['Close'].squeeze(), index=out.index)
    high  = pd.Series(out['High'].squeeze(),  index=out.index) if 'High' in out else close
    low   = pd.Series(out['Low'].squeeze(),   index=out.index) if 'Low' in out else close

    close = pd.to_numeric(close, errors='coerce')
    high  = pd.to_numeric(high,  errors='coerce')
    low   = pd.to_numeric(low,   errors='coerce')

    out['MA9'] = close.rolling(9).mean()
    out['RSI14'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd_obj = ta.trend.MACD(close=close)
    out['MACD'] = macd_obj.macd()
    out['MACD_signal'] = macd_obj.macd_signal()
    # ATR untuk risk plan
    try:
        atr_obj = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
        out['ATR14'] = atr_obj.average_true_range()
    except Exception:
        out['ATR14'] = np.nan
    # Volume MA20
    if 'Volume' in out:
        out['VolMA20'] = out['Volume'].rolling(20).mean()
    return out

def simple_signal(row):
    try:
        c=row['Close']; ma=row['MA9']; rsi=row['RSI14']; macd=row['MACD']; sig=row['MACD_signal']
        if pd.isna([c,ma,rsi,macd,sig]).any():
            return "WAIT"
        if c>ma and rsi<70 and macd>sig: return "BUY"
        if c<ma and rsi>50 and macd<sig: return "SELL"
        return "HOLD"
    except Exception:
        return "WAIT"

def sr_levels_lastbar(df: pd.DataFrame):
    last_h=float(df['High'].iloc[-1]); last_l=float(df['Low'].iloc[-1]); last_c=float(df['Close'].iloc[-1])
    pivot=(last_h+last_l+last_c)/3.0
    r1=(2*pivot)-last_l; s1=(2*pivot)-last_h
    r2=pivot+(last_h-last_l); s2=pivot-(last_h-last_l)
    return pivot,r1,r2,s1,s2

def compute_entry_tp_sl(df: pd.DataFrame, swing_window:int=10):
    c=float(df['Close'].iloc[-1])
    pivot,r1,r2,s1,s2 = sr_levels_lastbar(df)
    swing_res=float(df['High'].tail(swing_window).max())
    swing_sup=float(df['Low'].tail(swing_window).min())
    atr=float(df['ATR14'].iloc[-1]) if 'ATR14' in df and not pd.isna(df['ATR14'].iloc[-1]) else c*0.02

    # Entry BoW: level terdekat di bawah harga
    candidates=[lvl for lvl in [df['MA9'].iloc[-1], pivot, s1, swing_sup] if not pd.isna(lvl) and lvl<=c]
    entry = max(candidates) if candidates else c*0.99
    # TP: resistance terdekat di atas harga
    res=[lvl for lvl in [r1,r2,swing_res] if lvl>=c]
    tp=min(res) if res else c+1.5*atr
    # SL: support terdekat di bawah harga
    sup=[lvl for lvl in [s1,s2,swing_sup] if lvl<=c]
    sl=max(sup) if sup else c-1.0*atr

    # jaga R:R >= 1
    risk=c-sl; reward=tp-c
    if risk>0 and (reward/risk)<1.0:
        tp=c+risk*1.2
    return float(entry), float(tp), float(sl), (pivot,r1,r2,s1,s2, swing_res, swing_sup)

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="Stock Screener + Portfolio", layout="wide")
st.title("📊 Stock Screener + Portfolio (Global/BEI)")

with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Screener")
    default_tickers = "IKAN.JK, BRMS.JK, SIDO.JK, ANTM.JK, TLKM.JK, CPIN.JK, PGAS.JK, MDKA.JK, ADRO.JK"
    tickers_text = st.text_area("Daftar ticker (pisahkan koma). Contoh BEI pakai .JK", default_tickers)
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    period = st.selectbox("Periode data", ["1mo","3mo","6mo","1y"], index=1)
    if st.button("🔍 Screening"):
        st.session_state["run_screen"] = True

# --------- Screener ----------
if st.session_state.get("run_screen", False):
    results=[]
    for t in tickers:
        df = get_ohlcv(t, period=period, interval="1d")
        if df is None or df.empty:
            results.append({"Ticker": t, "Close": None, "MA9": None, "RSI14": None,
                            "MACD": None, "MACD_signal": None, "Rekomendasi": "NO DATA"})
            continue
        dfi = compute_indicators(df)
        last = dfi.iloc[-1]
        results.append({
            "Ticker": t,
            "Close": round(float(last['Close']),2),
            "MA9": round(float(last['MA9']),2) if not pd.isna(last['MA9']) else None,
            "RSI14": round(float(last['RSI14']),2) if not pd.isna(last['RSI14']) else None,
            "MACD": round(float(last['MACD']),3) if not pd.isna(last['MACD']) else None,
            "MACD_signal": round(float(last['MACD_signal']),3) if not pd.isna(last['MACD_signal']) else None,
            "Rekomendasi": simple_signal(last)
        })
    screener_df = pd.DataFrame(results)

    def color_sig(s):
        styles=[]
        for v in s:
            if v=="BUY": styles.append("background-color:#2ecc71;color:black;font-weight:bold")
            elif v=="SELL": styles.append("background-color:#e74c3c;color:white;font-weight:bold")
            elif v=="HOLD": styles.append("background-color:#f1c40f;color:black;font-weight:bold")
            else: styles.append("")
        return styles

    st.subheader("🧾 Hasil Screening")
    st.dataframe(screener_df.style.apply(color_sig, subset=["Rekomendasi"]), use_container_width=True)
    st.download_button("📥 Download CSV", screener_df.to_csv(index=False).encode("utf-8"), "screener_result.csv")

# --------- Analisis & Portofolio detail ----------
st.markdown("---")
st.subheader("🔎 Analisis Detail + Portofolio")

colA, colB, colC = st.columns([2,1,1])
with colA:
    sel_ticker = st.text_input("Ticker untuk analisis detail (contoh: TLKM.JK atau AAPL)", (tickers[0] if tickers else "TLKM.JK")).upper()
with colB:
    avg_buy = st.number_input("Avg Buy (opsional)", min_value=0.0, value=0.0, step=0.01)
with colC:
    lot = st.number_input("Lot (opsional)", min_value=0, value=0, step=1)

dfd = get_ohlcv(sel_ticker, period="6mo", interval="1d")
if dfd is None or dfd.empty:
    st.warning(f"Tidak ada data untuk {sel_ticker}.")
else:
    dfi = compute_indicators(dfd)
    last = dfi.iloc[-1]
    last_c=float(last['Close'])
    last_ma=float(last['MA9']) if not pd.isna(last['MA9']) else np.nan
    last_rsi=float(last['RSI14']) if not pd.isna(last['RSI14']) else np.nan
    last_macd=float(last['MACD']) if not pd.isna(last['MACD']) else np.nan
    last_sig=float(last['MACD_signal']) if not pd.isna(last['MACD_signal']) else np.nan

    entry,tp,sl,levels = compute_entry_tp_sl(dfi, swing_window=10)
    pivot,r1,r2,s1,s2, swing_res, swing_sup = levels

    # Rekomendasi
    rekom = simple_signal(last)

    # P/L portofolio
    if avg_buy>0 and lot>0:
        shares = lot*100
        modal = avg_buy*shares
        nilai = last_c*shares
        pnl   = nilai - modal
        pnl_pct = (pnl/modal*100) if modal>0 else 0.0
    else:
        pnl = pnl_pct = modal = nilai = shares = 0

    # Kartu ringkas
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Harga Sekarang", f"{last_c:.2f}")
    m2.metric("MA9", f"{last_ma:.2f}" if not np.isnan(last_ma) else "-")
    m3.metric("RSI14", f"{last_rsi:.2f}" if not np.isnan(last_rsi) else "-")
    m4.metric("Sinyal", rekom)

    st.write(f"**Harga Beli Ideal (BoW)**: **{entry:.2f}** | **TP**: **{tp:.2f}** | **SL**: **{sl:.2f}**")
    st.caption(f"Pivot: {pivot:.2f} | R1: {r1:.2f} | R2: {r2:.2f} | S1: {s1:.2f} | S2: {s2:.2f} | Swing High(10d): {swing_res:.2f} | Swing Low(10d): {swing_sup:.2f}")

    if shares>0:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Avg Buy", f"{avg_buy:.2f}")
        c2.metric("Qty (lembar)", f"{shares}")
        c3.metric("P/L (Rp)", f"{pnl:,.0f}")
        c4.metric("P/L (%)", f"{pnl_pct:.2f}%")

    # Chart
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55,0.2,0.25], vertical_spacing=0.04,
                        subplot_titles=(f"{sel_ticker} - Harga & MA9 (+Entry/TP/SL)", "RSI(14)", "MACD"))

    fig.add_trace(go.Candlestick(x=dfi.index, open=dfi['Open'], high=dfi['High'], low=dfi['Low'], close=dfi['Close'], name="Harga"), row=1,col=1)
    fig.add_trace(go.Scatter(x=dfi.index, y=dfi['MA9'], name="MA9", mode="lines", line=dict(color="orange")), row=1,col=1)

    # garis level
    for y, txt, col in [(entry,"Entry", "#3498db"), (tp,"TP","#2ecc71"), (sl,"SL","#e74c3c"),
                        (pivot,"Pivot","#7f8c8d"), (r1,"R1","#e67e22"), (r2,"R2","#d35400"),
                        (s1,"S1","#27ae60"), (s2,"S2","#16a085"),
                        (swing_res,f"Swing High","#c0392b"), (swing_sup,f"Swing Low","#2980b9")]:
        fig.add_hline(y=y, line_dash="dot", line_color=col, annotation_text=txt, row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=dfi.index, y=dfi['RSI14'], name="RSI14", mode="lines", line=dict(color="yellow")), row=2,col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2,col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2,col=1)

    # MACD
    fig.add_trace(go.Scatter(x=dfi.index, y=dfi['MACD'], name="MACD", mode="lines", line=dict(color="cyan")), row=3,col=1)
    fig.add_trace(go.Scatter(x=dfi.index, y=dfi['MACD_signal'], name="Signal", mode="lines", line=dict(color="magenta")), row=3,col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="white", row=3,col=1)

    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=900, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # Alasan indikator ringkas
    notes=[]
    notes.append(f"MA9: {last_ma:.2f} → harga {'di atas' if last_c>last_ma else 'di bawah'} MA9")
    if last_rsi<30: notes.append(f"RSI {last_rsi:.2f} → oversold (potensi rebound)")
    elif last_rsi>70: notes.append(f"RSI {last_rsi:.2f} → overbought (rawan koreksi)")
    else: notes.append(f"RSI {last_rsi:.2f} → netral")
    notes.append("MACD bullish" if last_macd>last_sig else "MACD bearish")
    st.markdown("**Alasan singkat:**")
    for n in notes: st.write(f"- {n}")
