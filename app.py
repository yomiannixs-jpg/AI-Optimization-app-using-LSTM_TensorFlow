import streamlit as st

st.set_page_config(page_title="NGX Pro Terminal", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
h1, h2, h3 {letter-spacing: 0.2px;}
div[data-testid="stMetric"] {border: 1px solid rgba(49,51,63,0.15); padding: 12px; border-radius: 12px;}
section[data-testid="stSidebar"] {border-right: 1px solid rgba(49,51,63,0.15);}
</style>
""",
    unsafe_allow_html=True,
)

st.title("NGX Pro Terminal")
st.caption("NGX Optimizer + Market Terminal (Twelve Data) + Strategy Lab.")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### NGX Optimizer")
    st.write("Upload NGX Excel (multi-sheet) or any CSV/XLSX. Run optimisation, get charts/tables/explanations, download ZIP.")
    st.page_link("pages/01_ngx_optimizer.py", label="Open NGX Optimizer →")
with c2:
    st.markdown("### Market Terminal")
    st.write("Pull historical/near-live OHLCV from Twelve Data by ticker and interval. Candles displayed in a trading-terminal style.")
    st.page_link("pages/02_market_terminal.py", label="Open Market Terminal →")
with c3:
    st.markdown("### Strategy Lab")
    st.write("Upload any dataset and run analytics (equity curves, rolling vol, correlations, drawdowns).")
    st.page_link("pages/03_strategy_lab.py", label="Open Strategy Lab →")

st.info("On Render: set Environment variable **TWELVEDATA_API_KEY** to enable market data. LSTM runs when TensorFlow is installed (this bundle includes TensorFlow).")
