import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from twelvedata import TDClient

st.set_page_config(page_title="Market Terminal", layout="wide")
st.title("Market Terminal (Twelve Data)")
st.caption("Trading-desk style market terminal: OHLCV + indicators (SMA/EMA/RSI/MACD) computed locally from the pulled data.")

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n//2)).mean()

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=max(2, n//2)).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

api_key_env = os.getenv("TWELVEDATA_API_KEY", "")

with st.sidebar:
    st.header("Connection")
    api_key = st.text_input("Twelve Data API key", value=api_key_env, type="password")
    st.caption("On Render: set TWELVEDATA_API_KEY in Environment.")
    st.divider()

    st.header("Instrument")
    symbol = st.text_input("Symbol", value="AAPL", help="Examples: AAPL, MSFT, TSLA, SPY, BTC/USD, EUR/USD")
    interval = st.selectbox("Interval", ["1min","5min","15min","30min","1h","4h","1day","1week","1month"], index=6)
    outputsize = st.number_input("Bars (outputsize)", 50, 5000, 700, 50)
    st.divider()

    st.header("Indicators")
    show_ma = st.checkbox("Show Moving Averages", value=True)
    ma_fast = st.number_input("MA fast", 5, 200, 20, 1)
    ma_slow = st.number_input("MA slow", 5, 400, 50, 1)

    show_rsi = st.checkbox("Show RSI", value=True)
    rsi_n = st.number_input("RSI period", 5, 50, 14, 1)

    show_macd = st.checkbox("Show MACD", value=True)
    macd_fast = st.number_input("MACD fast", 5, 30, 12, 1)
    macd_slow = st.number_input("MACD slow", 10, 60, 26, 1)
    macd_signal = st.number_input("MACD signal", 3, 30, 9, 1)

    st.divider()
    run = st.button("Load market data", type="primary", use_container_width=True)

if not run:
    st.info("Enter a symbol and click **Load market data**.")
    st.stop()

if not api_key:
    st.error("Missing Twelve Data API key.")
    st.stop()

td = TDClient(apikey=api_key)

with st.spinner("Fetching time series…"):
    try:
        ts = td.time_series(symbol=symbol, interval=interval, outputsize=int(outputsize), format="JSON")
        df = ts.as_pandas()
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.stop()

if df is None or df.empty:
    st.error("No data returned.")
    st.stop()

df = df.copy()
if "datetime" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")

for c in ["open","high","low","close","volume"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["open","high","low","close"], how="any")
if df.empty:
    st.error("Returned data has no usable OHLC rows after cleaning.")
    st.stop()

last = float(df["close"].iloc[-1])
prev = float(df["close"].iloc[-2]) if len(df) > 1 else last
chg = (last/prev - 1.0)*100 if prev else 0.0
rng = float(df["high"].iloc[-1] - df["low"].iloc[-1])

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Last", f"{last:,.4f}")
m2.metric("Change", f"{chg:,.2f}%")
m3.metric("Bars", f"{len(df):,}")
m4.metric("Range (last)", f"{rng:,.4f}")
m5.metric("Volume (last)", f"{float(df['volume'].iloc[-1]):,.0f}" if "volume" in df.columns and df["volume"].notna().any() else "—")

close = df["close"]
if show_ma:
    df["MA_fast"] = sma(close, int(ma_fast))
    df["MA_slow"] = sma(close, int(ma_slow))
if show_rsi:
    df["RSI"] = rsi(close, int(rsi_n))
if show_macd:
    macd_line, signal_line, hist = macd(close, int(macd_fast), int(macd_slow), int(macd_signal))
    df["MACD"] = macd_line
    df["MACD_signal"] = signal_line
    df["MACD_hist"] = hist

tab1, tab2, tab3 = st.tabs(["Chart", "Indicators tables", "Raw data"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price"
    ))
    if show_ma:
        fig.add_trace(go.Scatter(x=df.index, y=df["MA_fast"], mode="lines", name=f"SMA {int(ma_fast)}"))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA_slow"], mode="lines", name=f"SMA {int(ma_slow)}"))
    fig.update_layout(height=560, margin=dict(l=20,r=20,t=35,b=20), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if show_rsi and "RSI" in df.columns:
            rfig = go.Figure()
            rfig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"))
            rfig.add_hline(y=70)
            rfig.add_hline(y=30)
            rfig.update_layout(height=280, margin=dict(l=20,r=20,t=30,b=20), title="RSI")
            st.plotly_chart(rfig, use_container_width=True)
        else:
            st.info("Enable RSI in the sidebar to show the RSI panel.")
    with c2:
        if show_macd and "MACD" in df.columns:
            mfig = go.Figure()
            mfig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"))
            mfig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], mode="lines", name="Signal"))
            mfig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist"))
            mfig.update_layout(height=280, margin=dict(l=20,r=20,t=30,b=20), title="MACD")
            st.plotly_chart(mfig, use_container_width=True)
        else:
            st.info("Enable MACD in the sidebar to show the MACD panel.")

with tab2:
    st.subheader("Latest indicator readings")
    cols = ["close"]
    if show_ma: cols += ["MA_fast","MA_slow"]
    if show_rsi: cols += ["RSI"]
    if show_macd: cols += ["MACD","MACD_signal","MACD_hist"]
    st.dataframe(df[cols].tail(60), use_container_width=True)

with tab3:
    st.dataframe(df.tail(500), use_container_width=True)

st.caption("Source: Twelve Data Time Series API. Indicators are computed locally from the returned OHLCV.")
