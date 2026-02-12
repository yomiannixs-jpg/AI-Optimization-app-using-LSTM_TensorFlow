from __future__ import annotations

import os
import json
import math
import zipfile
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import layers, models  # type: ignore
    TENSORFLOW_OK = True
except Exception:
    TENSORFLOW_OK = False


def load_ngx_multisheet_xlsx(xlsx_path: str) -> pd.DataFrame:
    """Load multi-sheet XLSX: each sheet = one sector series.

    Each sheet should contain a Date column + a price column (any numeric column).
    Returns wide DataFrame indexed by Date with columns = sheet names.
    """
    xl = pd.ExcelFile(xlsx_path)
    try:
        _ = xl.sheet_names
    except Exception:
        pass
    series = {}
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        if df.shape[1] < 2:
            continue

        date_col = None
        for c in df.columns:
            if "date" in str(c).lower():
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)

        candidates = [c for c in df.columns if c != date_col]
        price_col = None
        for c in candidates:
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().mean() > 0.7:
                price_col = c
                df[c] = x
                break
        if price_col is None:
            continue

        s = (
            df.set_index(date_col)[price_col]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .bfill()
        )
        if s.notna().mean() < 0.7:
            continue
        series[sheet] = s

    if len(series) < 2:
        raise ValueError("Need >=2 usable sector series across sheets.")

    wide = pd.concat(series, axis=1).sort_index().ffill().bfill()
    keep = [c for c in wide.columns if wide[c].notna().mean() > 0.8]
    wide = wide[keep]
    if wide.shape[1] < 2:
        raise ValueError("After cleaning, fewer than 2 sector series remain.")
    try:
        xl.close()
    except Exception:
        pass
    return wide


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    rets = np.log(prices).diff().replace([np.inf, -np.inf], np.nan)
    return rets.dropna(how="all")


def equity_curve_from_log_returns(log_r: pd.Series) -> pd.Series:
    log_r = pd.Series(log_r).fillna(0.0)
    return np.exp(log_r.cumsum())


def drawdown_from_equity_curve(v: pd.Series) -> pd.Series:
    v = pd.Series(v).dropna()
    peak = v.cummax()
    return v / peak - 1.0


def annualize_return_from_log_returns(log_r: np.ndarray, freq=252) -> float:
    x = np.asarray(log_r, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")
    return float(np.exp(np.mean(x) * freq) - 1.0)


def annualize_vol_from_log_returns(log_r: np.ndarray, freq=252) -> float:
    x = np.asarray(log_r, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")
    return float(np.std(x) * np.sqrt(freq))


def sharpe_from_log_returns(log_r: np.ndarray, rf=0.0, freq=252) -> float:
    x = np.asarray(log_r, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")
    ex = x - rf / freq
    sd = np.std(ex)
    if sd < 1e-12:
        return float("nan")
    return float(np.mean(ex) / sd * np.sqrt(freq))


def compute_market_vol_daily(returns: pd.DataFrame, vol_window=21) -> pd.Series:
    vol_per_sector = returns.rolling(vol_window, min_periods=max(10, vol_window // 2)).std()
    return vol_per_sector.mean(axis=1).dropna()


def make_weekly_features(returns: pd.DataFrame) -> pd.DataFrame:
    weekly_sector_ret = returns.resample("W-FRI").sum(min_count=3)
    weekly_return_cs = weekly_sector_ret.mean(axis=1)

    mkt_vol_daily = compute_market_vol_daily(returns, vol_window=21)
    mkt_vol_weekly_last = mkt_vol_daily.resample("W-FRI").last()
    weekly_vol_mean = mkt_vol_daily.resample("W-FRI").mean()

    mom_8w = weekly_return_cs.rolling(8, min_periods=4).mean()
    rev_2w = -1.0 * weekly_return_cs.rolling(2, min_periods=1).mean()

    return pd.DataFrame({
        "weekly_return_cs": weekly_return_cs,
        "weekly_vol_mean": weekly_vol_mean,
        "mom_8w": mom_8w,
        "rev_2w": rev_2w,
        "mkt_vol_weekly": mkt_vol_weekly_last,
    }).dropna()


def make_regime_labels(mkt_vol_weekly: pd.Series, train_end_date: pd.Timestamp,
                       q_low=33, q_high=66) -> Tuple[pd.Series, float, float]:
    train_series = mkt_vol_weekly.loc[:train_end_date].dropna()
    if len(train_series) < 30:
        raise ValueError("Not enough weekly data to compute thresholds.")
    t1 = float(np.percentile(train_series.values, q_low))
    t2 = float(np.percentile(train_series.values, q_high))

    def lab(v: float) -> int:
        if v <= t1:
            return 0
        if v <= t2:
            return 1
        return 2

    y = mkt_vol_weekly.apply(lambda v: lab(float(v))).rename("regime")
    return y, t1, t2


def _make_lstm_sequences(X: pd.DataFrame, y: pd.Series, lookback=12):
    y = y.reindex(X.index).dropna()
    X = X.loc[y.index]

    Xv, yv, idx = [], [], []
    for i in range(lookback, len(X)):
        Xv.append(X.iloc[i - lookback:i].values)
        yv.append(int(y.iloc[i]))
        idx.append(X.index[i])
    return np.array(Xv), np.array(yv), pd.DatetimeIndex(idx)


def train_lstm_regime_classifier(feats_weekly: pd.DataFrame,
                                 y_regime: pd.Series,
                                 lookback=12,
                                 train_frac=0.8,
                                 epochs=20,
                                 batch_size=32,
                                 seed=42):
    if not TENSORFLOW_OK:
        return None, None, {"note": "TensorFlow not available; using rule-based regimes."}, None

    tf.random.set_seed(seed)
    np.random.seed(seed)

    Xv_raw, yv, seq_idx = _make_lstm_sequences(feats_weekly, y_regime, lookback=lookback)
    if len(Xv_raw) < 60:
        return None, None, {"note": "Not enough sequences for LSTM; using rule-based regimes."}, None

    split = int(train_frac * len(Xv_raw))
    Xtr_raw, Xte_raw = Xv_raw[:split], Xv_raw[split:]
    ytr, yte = yv[:split], yv[split:]

    mu = Xtr_raw.reshape(-1, Xtr_raw.shape[-1]).mean(axis=0)
    sd = Xtr_raw.reshape(-1, Xtr_raw.shape[-1]).std(axis=0) + 1e-8

    def norm(A):
        return (A - mu) / sd

    Xtr = norm(Xtr_raw)
    Xte = norm(Xte_raw)
    Xall = norm(Xv_raw)

    model = models.Sequential([
        layers.Input(shape=(Xtr.shape[1], Xtr.shape[2])),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=epochs, batch_size=batch_size, verbose=0)

    probs_all = model.predict(Xall, verbose=0)
    pred_all = probs_all.argmax(axis=1)

    yhat_te = model.predict(Xte, verbose=0).argmax(axis=1)
    acc = float((yhat_te == yte).mean())

    cm = np.zeros((3, 3), dtype=int)
    for yt, yh in zip(yte, yhat_te):
        cm[int(yt), int(yh)] += 1

    metrics = {"accuracy": acc, "confusion_matrix": cm.tolist()}

    probs_df = pd.DataFrame(probs_all, index=seq_idx, columns=["P_Low", "P_Med", "P_High"])
    pred_s = pd.Series(pred_all, index=seq_idx, name="pred_regime")
    return probs_df, pred_s, metrics, model


def shrink_covariance(cov: np.ndarray, lam: float = 0.10) -> np.ndarray:
    diag = np.diag(np.diag(cov))
    return (1 - lam) * cov + lam * diag


def _project_simplex_with_cap(w: np.ndarray, cap: float = 0.30) -> np.ndarray:
    w = np.clip(w, 0, cap)
    s = float(w.sum())
    if s <= 1e-12:
        return np.ones_like(w) / len(w)
    w = w / s
    for _ in range(30):
        over = w > cap
        if not over.any():
            break
        w[over] = cap
        rem = 1.0 - float(w.sum())
        if rem <= 1e-12:
            w = w / (float(w.sum()) + 1e-12)
            break
        under = ~over
        if under.sum() == 0:
            break
        w[under] += rem * (w[under] / (float(w[under].sum()) + 1e-12))
    return w


def mean_variance_weights(mu: np.ndarray, cov: np.ndarray, risk_aversion: float = 3.0, cap: float = 0.30) -> np.ndarray:
    n = cov.shape[0]
    cov = cov + 1e-8 * np.eye(n)
    inv = np.linalg.pinv(cov)
    w = (inv @ mu.reshape(-1, 1)).flatten() / (risk_aversion + 1e-12)
    return _project_simplex_with_cap(w, cap=cap)


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    cov = cov + 1e-8 * np.eye(n)
    ones = np.ones((n, 1))
    inv = np.linalg.pinv(cov)
    w = (inv @ ones)
    w = (w / (ones.T @ w)).flatten()
    w = np.clip(w, 0, None)
    return w / (float(w.sum()) + 1e-12)


def risk_parity_weights(cov: np.ndarray, iters=800, lr=0.05, tol=1e-6) -> np.ndarray:
    n = cov.shape[0]
    cov = cov + 1e-8 * np.eye(n)
    w = np.ones(n) / n
    for _ in range(iters):
        mrc = cov @ w
        rc = w * mrc
        target = float(rc.mean())
        grad = rc - target
        w = w - lr * grad
        w = np.clip(w, 0, None)
        w = w / (float(w.sum()) + 1e-12)
        if float(np.linalg.norm(grad)) < tol:
            break
    return w


@dataclass
class BacktestConfig:
    rebalance_freq: str = "W-FRI"
    estimation_window: int = 60
    cov_shrink_lam: float = 0.10
    mv_cap: float = 0.30
    mv_risk_aversion: float = 3.0
    tc_bps: float = 30.0
    freq: int = 252


def backtest_regime_conditioned(log_returns: pd.DataFrame, pred_regime_weekly: pd.Series, cfg: BacktestConfig):
    rets = log_returns.dropna(how="all").copy()
    if rets.empty:
        return pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=int)

    period_end = rets.resample(cfg.rebalance_freq).last().index
    period_end = period_end.intersection(rets.index)

    port_log_rets, dates = [], []
    weights_toggle, regime_toggle = [], []

    w_prev = np.ones(rets.shape[1]) / rets.shape[1]
    pred_regime_weekly = pred_regime_weekly.copy()
    pred_regime_weekly.index = pd.to_datetime(pred_regime_weekly.index)

    for t in range(cfg.estimation_window, len(rets)):
        date = rets.index[t]
        if date not in period_end:
            continue

        if date in pred_regime_weekly.index:
            regime = int(pred_regime_weekly.loc[date])
        else:
            prev = pred_regime_weekly.loc[:date]
            if prev.empty:
                continue
            regime = int(prev.iloc[-1])

        window = rets.iloc[t - cfg.estimation_window:t].dropna(how="all")
        if window.empty:
            continue

        mu = np.nanmean(window.values, axis=0)
        cov = np.cov(window.values.T)
        cov = shrink_covariance(cov, lam=cfg.cov_shrink_lam)

        if regime == 0:
            w = mean_variance_weights(mu, cov, risk_aversion=cfg.mv_risk_aversion, cap=cfg.mv_cap)
            policy = "Mean-Variance"
        elif regime == 1:
            w = risk_parity_weights(cov)
            policy = "Risk-Parity"
        else:
            w = min_variance_weights(cov)
            policy = "Minimum-Variance"

        pos = period_end.get_loc(date)
        if pos == len(period_end) - 1:
            next_slice = rets.loc[date:].iloc[1:]
        else:
            next_date = period_end[pos + 1]
            next_slice = rets.loc[date:next_date].iloc[1:]
        if next_slice.empty:
            continue

        tc_simple = (cfg.tc_bps / 10000.0) * float(np.sum(np.abs(w - w_prev)))
        tc_log = math.log(max(1.0 - tc_simple, 1e-9))
        tc_log_per_day = tc_log / max(len(next_slice), 1)

        period_port_log = (next_slice.values @ w)
        period_port_log_adj = period_port_log + tc_log_per_day

        for d, r in zip(next_slice.index, period_port_log_adj):
            dates.append(d)
            port_log_rets.append(float(r))

        weights_toggle.append((date, policy, *w))
        regime_toggle.append((date, regime))
        w_prev = w.copy()

    port_series = pd.Series(port_log_rets, index=pd.DatetimeIndex(dates)).sort_index()
    weights_df = pd.DataFrame(weights_toggle, columns=["Date", "Policy"] + list(log_returns.columns)).set_index("Date")
    regime_s = pd.Series({d: r for d, r in regime_toggle}, name="applied_regime").sort_index()
    return port_series, weights_df, regime_s


def _save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def make_plots(outdir: str, port_log: pd.Series, weights_df: pd.DataFrame, applied_regime: pd.Series):
    os.makedirs(outdir, exist_ok=True)

    v = equity_curve_from_log_returns(port_log)
    plt.figure(figsize=(10, 4))
    plt.plot(v.index, v.values)
    plt.title("Cumulative Returns (Equity Curve from Log Returns)")
    plt.xlabel("Date")
    plt.ylabel("Equity (start=1.0)")
    _save_fig(os.path.join(outdir, "equity_curve.png"))

    dd = drawdown_from_equity_curve(v)
    plt.figure(figsize=(10, 4))
    plt.plot(dd.index, dd.values)
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    _save_fig(os.path.join(outdir, "drawdown.png"))

    if not weights_df.empty:
        w = weights_df.drop(columns=["Policy"], errors="ignore")
        plt.figure(figsize=(12, 4))
        plt.imshow(w.T.values, aspect="auto", interpolation="nearest")
        plt.yticks(range(len(w.columns)), w.columns)
        step = max(1, len(w.index) // 10)
        xticks = range(0, len(w.index), step)
        xlabels = [str(w.index[i].date()) for i in xticks]
        plt.xticks(xticks, xlabels, rotation=45, ha="right")
        plt.title("Portfolio Weights (Weekly Rebalance)")
        plt.colorbar()
        _save_fig(os.path.join(outdir, "weights_heatmap.png"))

    if not applied_regime.empty:
        plt.figure(figsize=(10, 3))
        plt.step(applied_regime.index, applied_regime.values, where="post")
        plt.yticks([0, 1, 2], ["Low", "Medium", "High"])
        plt.title("Applied Volatility Regime Over Time (Weekly Rebalance)")
        _save_fig(os.path.join(outdir, "applied_regime_over_time.png"))


def make_tables(outdir: str, port_log: pd.Series, weights_df: pd.DataFrame, applied_regime: pd.Series, cfg: BacktestConfig):
    os.makedirs(outdir, exist_ok=True)

    r = port_log.dropna()
    v = equity_curve_from_log_returns(r)
    summary = pd.DataFrame([{
        "Strategy": "Regime-Conditioned (Low=MV, Med=RP, High=MinVar)",
        "Ann.Return": annualize_return_from_log_returns(r.values, freq=cfg.freq),
        "Ann.Vol": annualize_vol_from_log_returns(r.values, freq=cfg.freq),
        "Sharpe": sharpe_from_log_returns(r.values, freq=cfg.freq),
        "MaxDD": float(drawdown_from_equity_curve(v).min()) if not v.empty else float("nan"),
    }])
    summary.to_csv(os.path.join(outdir, "summary_stats.csv"), index=False)

    if not weights_df.empty:
        weights_df.to_csv(os.path.join(outdir, "weights_weekly.csv"))

    if not applied_regime.empty:
        applied_regime.to_csv(os.path.join(outdir, "applied_regime_weekly.csv"))

        regime_daily = applied_regime.reindex(r.index).ffill().dropna()
        aligned = pd.DataFrame({"port_log": r}).join(regime_daily.rename("regime"), how="inner")
        rows = []
        for k, name in [(0, "Low"), (1, "Medium"), (2, "High")]:
            seg = aligned.loc[aligned["regime"] == k, "port_log"].dropna()
            if seg.empty:
                continue
            vseg = equity_curve_from_log_returns(seg)
            rows.append({
                "Regime": name,
                "Obs(Days)": int(seg.shape[0]),
                "Ann.Return": annualize_return_from_log_returns(seg.values, freq=cfg.freq),
                "Ann.Vol": annualize_vol_from_log_returns(seg.values, freq=cfg.freq),
                "Sharpe": sharpe_from_log_returns(seg.values, freq=cfg.freq),
                "MaxDD": float(drawdown_from_equity_curve(vseg).min()) if not vseg.empty else float("nan"),
            })
        pd.DataFrame(rows).to_csv(os.path.join(outdir, "performance_by_regime.csv"), index=False)


def write_explanations(outdir: str, log_returns: pd.DataFrame, thr_low: float, thr_high: float):
    os.makedirs(outdir, exist_ok=True)

    sector_rows = []
    for c in log_returns.columns:
        x = log_returns[c].dropna().values
        sector_rows.append({
            "Sector": c,
            "Ann.Return": annualize_return_from_log_returns(x),
            "Ann.Vol": annualize_vol_from_log_returns(x),
            "Sharpe": sharpe_from_log_returns(x),
        })
    sector_df = pd.DataFrame(sector_rows).sort_values("Sharpe", ascending=False)
    sector_df.to_csv(os.path.join(outdir, "sector_stats.csv"), index=False)

    lines: List[str] = []
    lines.append("NGX AI OPTIMIZATION OUTPUT EXPLANATION")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Regime thresholds (training-only):")
    lines.append(f"- Low/Medium threshold: {thr_low:.6f}")
    lines.append(f"- Medium/High threshold: {thr_high:.6f}")
    lines.append("")
    lines.append("Charts:")
    lines.append("- equity_curve.png: growth of $1 invested using log-compounded returns.")
    lines.append("- drawdown.png: peak-to-trough decline; more negative means deeper losses.")
    lines.append("- weights_heatmap.png: allocation across sectors each rebalance date.")
    lines.append("- applied_regime_over_time.png: weekly regime (Low/Med/High).")
    lines.append("")
    lines.append("Tables:")
    lines.append("- summary_stats.csv: annualized return/volatility, Sharpe, max drawdown.")
    lines.append("- performance_by_regime.csv: same metrics within each regime.")
    lines.append("- sector_stats.csv: per-sector return/volatility/Sharpe ranking.")
    lines.append("- weights_weekly.csv: exact weekly weights.")
    (pd.Series(lines).to_string(index=False)).encode()

    with open(os.path.join(outdir, "analysis_explanation.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    s_lines = []
    s_lines.append("SECTOR-BY-SECTOR INTERPRETATION")
    s_lines.append("=" * 78)
    s_lines.append("Top sectors by Sharpe:")
    top = sector_df.head(min(10, len(sector_df)))
    for _, row in top.iterrows():
        s_lines.append(f"- {row['Sector']}: Sharpe={row['Sharpe']:.2f}, Ann.Return={row['Ann.Return']:.2%}, Ann.Vol={row['Ann.Vol']:.2%}")
    s_lines.append("")
    s_lines.append("Lowest sectors by Sharpe:")
    bottom = sector_df.tail(min(10, len(sector_df)))
    for _, row in bottom.iterrows():
        s_lines.append(f"- {row['Sector']}: Sharpe={row['Sharpe']:.2f}, Ann.Return={row['Ann.Return']:.2%}, Ann.Vol={row['Ann.Vol']:.2%}")
    with open(os.path.join(outdir, "sector_by_sector_explanation.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(s_lines))


def make_outputs_bundle(outdir: str) -> str:
    zip_path = os.path.join(outdir, "outputs_bundle.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(outdir):
            for fn in files:
                if fn == "outputs_bundle.zip":
                    continue
                fp = os.path.join(root, fn)
                z.write(fp, arcname=os.path.relpath(fp, outdir))
    return zip_path
