import os
import time
import tempfile
import streamlit as st
import pandas as pd
import engine

st.set_page_config(page_title="NGX AI Optimization", layout="wide")
st.title("NGX AI Portfolio Optimization (Multi-Sheet Excel)")
st.caption("Upload a multi-sheet Excel file (each sheet = one sector/index price series). The app generates charts, tables, and plain-English interpretations, and lets you download a ZIP of all outputs.")

with st.sidebar:
    st.header("Upload")
    up = st.file_uploader("Upload multi-sheet XLSX", type=["xlsx"])

    st.header("Model options")
    use_lstm = st.checkbox("Use LSTM regime classifier (TensorFlow)", value=False)
    estimation_window = st.number_input("Estimation window (days)", 30, 252, 60, 5)
    tc_bps = st.number_input("Transaction costs (bps)", 0.0, 200.0, 30.0, 5.0)
    mv_cap = st.number_input("Mean–Variance cap", 0.05, 1.0, 0.30, 0.05)

    st.header("Outputs")
    show_all_graphs = st.checkbox("Show all graphs", value=True)
    show_all_tables = st.checkbox("Show all tables", value=True)
    show_explanations = st.checkbox("Show plain-English interpretations", value=True)

    st.divider()
    run_btn = st.button("Run optimisation", type="primary", use_container_width=True)

if up is None:
    st.info("Upload your NGX multi-sheet XLSX to begin.")
    st.stop()

tmp_path = None
try:
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(up.getbuffer())
        tmp.flush()
        tmp_path = tmp.name

    prices = engine.load_ngx_multisheet_xlsx(tmp_path)
finally:
    if tmp_path and os.path.exists(tmp_path):
        for _ in range(12):
            try:
                os.remove(tmp_path)
                break
            except PermissionError:
                time.sleep(0.2)
            except Exception:
                break

log_returns = engine.to_log_returns(prices)

st.subheader("Data preview")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Prices (tail)**")
    st.dataframe(prices.tail(10), use_container_width=True)
with c2:
    st.markdown("**Log returns (tail)**")
    st.dataframe(log_returns.tail(10), use_container_width=True)

if not run_btn:
    st.warning("Click **Run optimisation** in the sidebar to generate results.")
    st.stop()

with st.spinner("Running optimisation…"):
    feats_weekly = engine.make_weekly_features(log_returns)
    mkt_vol_weekly = feats_weekly["mkt_vol_weekly"]

    train_end = feats_weekly.index[int(0.8 * len(feats_weekly.index))]
    y_regime, thr_low, thr_high = engine.make_regime_labels(mkt_vol_weekly, train_end_date=train_end)

    st.subheader("Regime thresholds (training-only)")
    st.write({"Low/Medium": thr_low, "Medium/High": thr_high})

    pred_regime_weekly = y_regime.copy()
    reg_metrics = {"note": "Rule-based regimes used (TensorFlow/LSTM not used)."}
    probs_df = None

    if use_lstm:
        probs_df, pred_seq, metrics, _ = engine.train_lstm_regime_classifier(feats_weekly, y_regime)
        if pred_seq is not None:
            pred_regime_weekly = pred_seq
            reg_metrics = metrics
            st.success("LSTM regimes generated and used.")
        else:
            reg_metrics = metrics
            st.warning(metrics.get("note", "LSTM unavailable; using rule-based regimes instead."))

    st.subheader("Regime classifier diagnostics")
    st.json(reg_metrics)
    if probs_df is not None:
        st.markdown("**Predicted regime probabilities (tail)**")
        st.dataframe(probs_df.tail(12), use_container_width=True)

    cfg = engine.BacktestConfig(
        estimation_window=int(estimation_window),
        mv_cap=float(mv_cap),
        tc_bps=float(tc_bps),
    )

    port_log, weights_df, applied_regime = engine.backtest_regime_conditioned(log_returns, pred_regime_weekly, cfg)
    if port_log.empty:
        st.error("Backtest produced no results. Check your data formatting.")
        st.stop()

    outdir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(outdir, exist_ok=True)

    engine.make_plots(outdir, port_log, weights_df, applied_regime)
    engine.make_tables(outdir, port_log, weights_df, applied_regime, cfg)
    engine.write_explanations(outdir, log_returns, thr_low, thr_high)
    zip_path = engine.make_outputs_bundle(outdir)

st.success("Done. Results are ready below.")

if show_all_graphs:
    st.subheader("All graphs")
    pngs = sorted([p for p in os.listdir(outdir) if p.lower().endswith(".png")])
    if not pngs:
        st.info("No PNG charts were produced.")
    else:
        for i in range(0, len(pngs), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(pngs):
                    fn = pngs[i + j]
                    cols[j].image(os.path.join(outdir, fn), caption=fn)

if show_all_tables:
    st.subheader("All tables")
    csvs = sorted([p for p in os.listdir(outdir) if p.lower().endswith(".csv")])
    if not csvs:
        st.info("No CSV tables were produced.")
    else:
        for fn in csvs:
            st.markdown(f"**{fn}**")
            try:
                df = pd.read_csv(os.path.join(outdir, fn))
                st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Could not read {fn}: {e}")

if show_explanations:
    st.subheader("Plain-English interpretation")
    txts = sorted([p for p in os.listdir(outdir) if p.lower().endswith(".txt")])
    if not txts:
        st.info("No interpretation text files were produced.")
    else:
        for fn in txts:
            st.markdown(f"**{fn}**")
            try:
                with open(os.path.join(outdir, fn), "r", encoding="utf-8") as f:
                    st.text(f.read())
            except Exception as e:
                st.warning(f"Could not read {fn}: {e}")

st.subheader("Download all outputs")
with open(zip_path, "rb") as f:
    st.download_button(
        "Download outputs_bundle.zip",
        data=f,
        file_name="outputs_bundle.zip",
        mime="application/zip",
        use_container_width=True,
    )
