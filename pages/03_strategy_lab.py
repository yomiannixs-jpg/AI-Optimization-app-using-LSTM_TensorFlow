import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils_data import load_any_prices
import engine

st.set_page_config(page_title="Strategy Lab", layout="wide")
st.title("Strategy Lab")
st.caption("Upload any dataset and run core trading analytics.")

with st.sidebar:
    up = st.file_uploader("Upload XLSX/CSV", type=["xlsx","csv"])
    window = st.number_input("Rolling window (days)", 10, 252, 63, 5)
    run = st.button("Run analytics", type="primary", use_container_width=True)

if up is None:
    st.info("Upload a dataset to begin.")
    st.stop()

prices = load_any_prices(up.getvalue(), up.name)
rets = engine.to_log_returns(prices).dropna(how="all")

st.subheader("Preview")
c1, c2 = st.columns(2)
with c1:
    st.dataframe(prices.tail(8), use_container_width=True)
with c2:
    st.dataframe(rets.tail(8), use_container_width=True)

if not run:
    st.stop()

st.subheader("Equity curves (log-compounded)")
fig = go.Figure()
for col in rets.columns[:12]:
    v = np.exp(rets[col].fillna(0).cumsum())
    fig.add_trace(go.Scatter(x=v.index, y=v.values, mode="lines", name=str(col)))
fig.update_layout(height=420, margin=dict(l=20,r=20,t=30,b=20))
st.plotly_chart(fig, use_container_width=True)

st.subheader(f"Rolling volatility ({int(window)} days)")
fig2 = go.Figure()
for col in rets.columns[:12]:
    rv = rets[col].rolling(int(window)).std() * np.sqrt(252)
    fig2.add_trace(go.Scatter(x=rv.index, y=rv.values, mode="lines", name=str(col)))
fig2.update_layout(height=420, margin=dict(l=20,r=20,t=30,b=20))
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Correlation (last window)")
corr = rets.tail(int(window)).corr()
st.dataframe(corr.round(3), use_container_width=True)

st.subheader("Drawdowns (selected series)")
fig3 = go.Figure()
for col in rets.columns[:6]:
    v = np.exp(rets[col].fillna(0).cumsum())
    peak = pd.Series(v).cummax()
    dd = v/peak - 1
    fig3.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name=str(col)))
fig3.update_layout(height=420, margin=dict(l=20,r=20,t=30,b=20))
st.plotly_chart(fig3, use_container_width=True)
