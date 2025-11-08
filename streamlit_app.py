# streamlit_app.py
import warnings
warnings.filterwarnings("ignore")

import os, random, time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, average_precision_score
)

# Optional ML libs (safe if not installed)
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

# Data sources
import yfinance as yf
try:
    from nsepython import nse_fetch  # fallback source
except Exception:
    nse_fetch = None

# -------------------------------
# Page config & reproducibility
# -------------------------------
st.set_page_config(page_title="Indian Equity Direction ML — RELIANCE", layout="wide")

GLOBAL_SEED = 42
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# -------------------------------
# Helpers
# -------------------------------
def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-index columns and standardize names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in t if str(x)]).strip() for t in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df

def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Yahoo/NSE frames to Open/High/Low/Close/Volume index by date."""
    df = _flatten_cols(df.dropna(how="all").sort_index())
    def pick(key, avoid=None):
        cols = [c for c in df.columns if key in c]
        if avoid:
            filtered = [c for c in cols if avoid not in c]
            if filtered:
                return filtered[0]
        return cols[0] if cols else None
    open_c  = pick("open")
    high_c  = pick("high")
    low_c   = pick("low")
    close_c = pick("close", avoid="adj")
    vol_c   = pick("volume")
    if close_c is None or vol_c is None:
        raise RuntimeError("Missing close/volume columns in response.")
    out = pd.DataFrame(index=df.index)
    if open_c and high_c and low_c:
        out["Open"] = pd.to_numeric(df[open_c], errors="coerce")
        out["High"] = pd.to_numeric(df[high_c], errors="coerce")
        out["Low"]  = pd.to_numeric(df[low_c],  errors="coerce")
    out["Close"]  = pd.to_numeric(df[close_c], errors="coerce")
    out["Volume"] = pd.to_numeric(df[vol_c],   errors="coerce")
    out = out.dropna(subset=["Close", "Volume"])
    if out.empty:
        raise RuntimeError("Data loaded, but rows dropped during cleanup.")
    return out

# -------------------------------
# Robust loader: Yahoo primary, NSE fallback
# -------------------------------
@st.cache_data(show_spinner=False)
def load_ohlcv(symbol: str, years: int) -> pd.DataFrame:
    """
    Load Indian OHLCV for last N years.
    Order: Yahoo (period→dates→history→raw) → NSE fallback (nsepython).
    """
    raw = symbol.strip()
    t = raw.upper()
    if (not t.startswith("^")) and (".NS" not in t) and (".BO" not in t):
        t = t + ".NS"

    start = (datetime.utcnow() - timedelta(days=365 * years + 30)).date().isoformat()
    end   = (datetime.utcnow() + timedelta(days=1)).date().isoformat()

    def _try(fn, attempts=3):
        for k in range(attempts):
            try:
                df = fn()
                if df is not None and len(df) > 0:
                    return df
            except Exception:
                pass
            time.sleep(0.6 * (k + 1))
        return None

    # A) Yahoo (period)
    df = _try(lambda: yf.download(t, period=f"{years}y", interval="1d",
                                  auto_adjust=False, progress=False, threads=False))
    # B) Yahoo (date range)
    if df is None:
        df = _try(lambda: yf.download(t, start=start, end=end, interval="1d",
                                      auto_adjust=False, progress=False, threads=False))
    # C) Yahoo Ticker.history
    if df is None:
        tk = yf.Ticker(t)
        df = _try(lambda: tk.history(start=start, end=end, interval="1d", auto_adjust=False))
    # D) Yahoo raw symbol (no .NS)
    if df is None and t != raw:
        df = _try(lambda: yf.download(raw, start=start, end=end, interval="1d",
                                      auto_adjust=False, progress=False, threads=False))

    if df is not None and len(df) > 0:
        try:
            return _clean_ohlcv(df)
        except Exception:
            pass

    # E) NSE fallback (no rate limits) — requires nsepython
    if nse_fetch is not None:
        sym = raw.replace(".NS", "").upper()
        start_dt = (datetime.utcnow() - timedelta(days=365 * years + 5))
        end_dt   = datetime.utcnow()
        url = (
            "https://www.nseindia.com/api/historical/cm/equity"
            f"?symbol={sym}&series=[%22EQ%22]"
            f"&from={start_dt.strftime('%d-%m-%Y')}"
            f"&to={end_dt.strftime('%d-%m-%Y')}"
        )
        try:
            nse_df = nse_fetch(url)
            if nse_df is not None and len(nse_df) > 0:
                rename_map = {
                    "CH_OPENING_PRICE": "Open",
                    "CH_TRADE_HIGH_PRICE": "High",
                    "CH_TRADE_LOW_PRICE": "Low",
                    "CH_CLOSING_PRICE": "Close",
                    "CH_TOT_TRADED_QTY": "Volume",
                    "CH_TIMESTAMP": "Date",
                }
                for k in rename_map:
                    if k not in nse_df.columns:
                        raise RuntimeError("NSE schema changed")
                nse_df = nse_df.rename(columns=rename_map)
                nse_df["Date"] = pd.to_datetime(nse_df["Date"], errors="coerce")
                nse_df = nse_df.dropna(subset=["Date"]).set_index("Date")
                nse_df = nse_df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna()
                nse_df = nse_df.sort_index()
                if len(nse_df) > 0:
                    return nse_df
        except Exception:
            pass

    raise RuntimeError("No data returned from Yahoo/NSE. Try later or another symbol.")

# -------------------------------
# Features
# -------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    o = df.copy()
    close = o["Close"]; vol = o["Volume"]
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    if isinstance(vol, pd.DataFrame):   vol   = vol.iloc[:, 0]

    ret = close.pct_change()
    o["Return"] = ret
    o["Volatility_10"] = ret.rolling(10, min_periods=10).std()
    o["Volatility_20"] = ret.rolling(20, min_periods=20).std()

    for w in [5, 10, 20, 50, 100]:
        o[f"SMA_{w}"] = close.rolling(w, min_periods=w).mean()
    for w in [5, 12, 26]:
        o[f"EMA_{w}"] = close.ewm(span=w, adjust=False).mean()

    o["Close_to_SMA_10"] = close.div(o["SMA_10"])
    o["Close_to_SMA_20"] = close.div(o["SMA_20"])

    ma20 = close.rolling(20, min_periods=20).mean()
    sd20 = close.rolling(20, min_periods=20).std()
    o["BB_Upper"] = ma20 + 2 * sd20
    o["BB_Lower"] = ma20 - 2 * sd20
    o["BB_Width"] = (o["BB_Upper"] - o["BB_Lower"]).div(ma20)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    o["MACD"] = macd
    o["MACD_Signal"] = macd.ewm(span=9, adjust=False).mean()
    o["MACD_Hist"] = o["MACD"] - o["MACD_Signal"]

    delta = close.diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / 14, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / 14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    o["RSI_14"] = 100 - (100 / (1 + rs))

    for lag in [1, 2, 3, 5]:
        o[f"Return_lag_{lag}"] = ret.shift(lag)

    o["Volume_Change"] = vol.pct_change()
    o["Vol_to_SMA20"] = vol.div(vol.rolling(20, min_periods=20).mean())

    # Target: next-day direction (1 = Up, 0 = Down)
    o["y"] = (close.shift(-1) > close).astype(int)

    return o.dropna().copy()

def compute_indicators_only(df: pd.DataFrame) -> pd.DataFrame:
    return compute_indicators(df).drop(columns=["y"])

def apply_feature_clipping(o: pd.DataFrame) -> pd.DataFrame:
    o = o.copy()
    if "Return" in o.columns:
        o["Return"] = o["Return"].clip(lower=-0.15, upper=0.15)
    if "Volatility_10" in o.columns:
        o["Volatility_10"] = o["Volatility_10"].clip(upper=0.20)
    if "Volatility_20" in o.columns:
        o["Volatility_20"] = o["Volatility_20"].clip(upper=0.20)
    if "Volume_Change" in o.columns:
        o["Volume_Change"] = o["Volume_Change"].clip(lower=-5, upper=5)
    if "Vol_to_SMA20" in o.columns:
        o["Vol_to_SMA20"] = o["Vol_to_SMA20"].clip(lower=0, upper=5)
    o = o.replace([np.inf, -np.inf], np.nan)
    return o

# -------------------------------
# Modeling
# -------------------------------
def build_models():
    preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
    ])

    models = {
        "LogisticRegression": Pipeline([
            ("prep", preprocess),
            ("scale", StandardScaler()),
            ("clf", __import__("sklearn.linear_model").linear_model.LogisticRegression(max_iter=2000, random_state=GLOBAL_SEED))
        ]),
        "RandomForest": Pipeline([
            ("prep", preprocess),
            ("clf", __import__("sklearn.ensemble").ensemble.RandomForestClassifier(random_state=GLOBAL_SEED, n_jobs=1))
        ]),
        "HistGradientBoosting": Pipeline([
            ("prep", preprocess),
            ("clf", __import__("sklearn.ensemble").ensemble.HistGradientBoostingClassifier(random_state=GLOBAL_SEED))
        ]),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline([
            ("prep", preprocess),
            ("clf", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=GLOBAL_SEED,
                n_estimators=300,
                learning_rate=0.05,
                nthread=1
            ))
        ])

    if CatBoostClassifier is not None:
        models["CatBoost"] = Pipeline([
            ("prep", preprocess),
            ("clf", CatBoostClassifier(
                verbose=0,
                random_seed=GLOBAL_SEED,
                iterations=300,
                learning_rate=0.05,
                depth=6,
                thread_count=1
            ))
        ])

    grids = {
        "LogisticRegression": {"clf__C": np.logspace(-2, 1, 6)},
        "RandomForest": {"clf__n_estimators": [150, 250, 350], "clf__max_depth": [6, 10, None]},
        "HistGradientBoosting": {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [None, 6]},
        "XGBoost": {"clf__max_depth": [3, 5], "clf__subsample": [0.8, 1.0]},
        "CatBoost": {"clf__depth": [4, 6], "clf__l2_leaf_reg": [1, 5]},
    }
    return models, grids

def tune_threshold_cv(best_estimator, X_train, y_train, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    probs = np.zeros_like(y_train, dtype=float)
    for tr, va in tscv.split(X_train):
        m = Pipeline(best_estimator.steps)  # shallow clone
        m.fit(X_train[tr], y_train[tr])
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(X_train[va])[:, 1]
        elif hasattr(m, "decision_function"):
            s = m.decision_function(X_train[va])
            p = 1 / (1 + np.exp(-s))
        else:
            p = m.predict(X_train[va]).astype(float)
        probs[va] = p
    ths = np.linspace(0.25, 0.75, 51)
    best_th, best_f1 = 0.5, -1
    for t in ths:
        pred = (probs >= t).astype(int)
        f1 = f1_score(y_train, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, t
    return float(best_th), float(best_f1)

def plot_cm_small(cm, title):
    fig = plt.figure(figsize=(5, 3.2))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = np.arange(2)
    plt.xticks(ticks, ["Down", "Up"]); plt.yticks(ticks, ["Down", "Up"])
    thr = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > thr else "black", fontsize=11)
    plt.ylabel("Actual"); plt.xlabel("Predicted"); plt.tight_layout()
    return fig

def plot_roc_small(y_true, prob, title):
    fpr, tpr, _ = roc_curve(y_true, prob)
    fig = plt.figure(figsize=(5, 3.2))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "--", linewidth=1)
    plt.title(title); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=9); plt.tight_layout()
    return fig

# -------------------------------
# Sidebar (auto-load; no buttons)
# -------------------------------
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker (Yahoo/NSE)", value="RELIANCE.NS")
years = st.sidebar.slider("Years of history", 2, 5, 2)
use_trend_filter = st.sidebar.checkbox("Trend Filter (SMA50 > SMA200)", value=True)

# -------------------------------
# AUTO LOAD (visible status + cache)
# -------------------------------
with st.status("Fetching data…", expanded=False) as s:
    try:
        df = load_ohlcv(ticker, years)
        feat = compute_indicators(df)
        feat = apply_feature_clipping(feat).replace([np.inf, -np.inf], np.nan).dropna()
        s.update(label=f"✅ Loaded {len(df)} rows for {ticker}", state="complete")
    except Exception as e:
        s.update(label=f"❌ Data load failed: {e}", state="error")
        st.stop()

data = feat.copy()
feature_cols = [c for c in data.columns if c != "y"]
X = data[feature_cols].values
y = data["y"].values
split_idx = int(0.8 * len(data))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# -------------------------------
# Tabs
# -------------------------------
tab_overview, tab_train, tab_eval, tab_backtest, tab_predict, tab_insights = st.tabs(
    ["Overview", "Train", "Evaluation", "Backtest", "Predict", "Insights"]
)

# -------------------------------
# Overview
# -------------------------------
with tab_overview:
    st.subheader("Price • Volume • RSI • MACD")

    range_choice = st.selectbox("Show price history for:", ["30D", "90D", "180D", "365D", "All"], index=2)
    def slice_by_range(df_in, choice):
        if choice == "All": return df_in
        days = int(choice[:-1]); return df_in.tail(days)
    df_view = slice_by_range(df, range_choice)

    have_ohlc = all(c in df_view.columns for c in ["Open", "High", "Low", "Close"])
    have_vol  = "Volume" in df_view.columns

    close = df_view["Close"].astype(float)
    delta = close.diff()
    up = delta.clip(lower=0.0); down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi14 = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.15, 0.15, 0.15],
                        vertical_spacing=0.03,
                        subplot_titles=("Price", "Volume", "RSI (14)", "MACD (12,26,9)"))

    if have_ohlc:
        fig.add_trace(go.Candlestick(x=df_view.index,
                                     open=df_view["Open"].astype(float),
                                     high=df_view["High"].astype(float),
                                     low=df_view["Low"].astype(float),
                                     close=df_view["Close"].astype(float),
                                     increasing_line_color="#16A34A",
                                     decreasing_line_color="#DC2626",
                                     name="Price"), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df_view.index, y=close, mode="lines", name="Close"), row=1, col=1)

    if have_vol:
        vol_colors = np.where(
            df_view["Close"].values >= (df_view["Open"].values if have_ohlc else df_view["Close"].shift(1).values),
            "#16A34A", "#DC2626"
        )
        fig.add_trace(go.Bar(x=df_view.index, y=df_view["Volume"].astype(float),
                             marker_color=vol_colors, opacity=0.7, name="Volume"), row=2, col=1)

    fig.add_trace(go.Scatter(x=df_view.index, y=rsi14, mode="lines", name="RSI(14)"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_width=1, row=3, col=1)

    fig.add_trace(go.Bar(x=df_view.index, y=macd_hist, name="MACD Hist", opacity=0.6), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_view.index, y=macd, mode="lines", name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_view.index, y=macd_signal, mode="lines", name="Signal"), row=4, col=1)

    fig.update_layout(height=900, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_rangeslider_visible=False, showlegend=False)
    fig.update_yaxes(title_text=f"{ticker} Price", row=1, col=1)
    if have_vol: fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Train (auto on first visit)
# -------------------------------
with tab_train:
    st.subheader("Model Training (RandomizedSearchCV, cv=3)")

    models, grids = build_models()
    model_names = list(models.keys())
    default_idx = model_names.index("HistGradientBoosting") if "HistGradientBoosting" in model_names else 0
    model_choice = st.selectbox("Choose model", model_names, index=default_idx)
    pipe = models[model_choice]
    tscv = TimeSeriesSplit(n_splits=3)

    if ("best_est" not in st.session_state) or (st.session_state.get("trained_model_name") != model_choice):
        with st.status("Training model…", expanded=False) as s:
            search = RandomizedSearchCV(
                pipe,
                param_distributions=grids[model_choice],
                n_iter=8, cv=tscv, scoring="f1", n_jobs=1,
                random_state=GLOBAL_SEED, refit=True
            )
            search.fit(X_train, y_train)
            th_cv, thcv_f1 = tune_threshold_cv(search.best_estimator_, X_train, y_train, n_splits=3)
            st.session_state.update({
                "best_est": search.best_estimator_,
                "best_params": search.best_params_,
                "best_cv_f1": float(search.best_score_),
                "cv_threshold": float(th_cv),
                "model_choice": model_choice,
                "trained_model_name": model_choice
            })
            s.update(label="✅ Training complete", state="complete")

    best_est = st.session_state["best_est"]
    th = st.session_state["cv_threshold"]
    st.success(f"{model_choice} — Best CV F1: {st.session_state['best_cv_f1']:.4f}")
    st.caption("Best hyperparameters:")
    st.json(st.session_state["best_params"])
    st.info(f"Auto-tuned threshold from CV: {th:.3f}")

# -------------------------------
# Evaluation
# -------------------------------
with tab_eval:
    st.subheader("Holdout Evaluation")

    best_est = st.session_state["best_est"]
    th = st.session_state["cv_threshold"]

    best_est.fit(X_train, y_train)
    if hasattr(best_est, "predict_proba"):
        y_proba = best_est.predict_proba(X_test)[:, 1]
    elif hasattr(best_est, "decision_function"):
        s = best_est.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-s))
    else:
        y_proba = best_est.predict(X_test).astype(float)
    y_pred = (y_proba >= th).astype(int)

    if use_trend_filter:
        close_series = data["Close"]
        sma50 = close_series.rolling(50).mean()
        sma200 = close_series.rolling(200).mean()
        trend = (sma50 > sma200).astype(int).iloc[split_idx:].values
        y_pred = y_pred * trend

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    cm  = confusion_matrix(y_test, y_pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Precision", f"{prec:.3f}")
    c3.metric("Recall", f"{rec:.3f}")
    c4.metric("F1", f"{f1:.3f}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.pyplot(plot_cm_small(cm, f"Confusion Matrix — {st.session_state.get('model_choice','Model')}"))
    with col_b:
        st.pyplot(plot_roc_small(y_test, y_proba, f"ROC Curve — {st.session_state.get('model_choice','Model')}"))

    st.session_state["y_pred_eval"] = y_pred

# -------------------------------
# Backtest
# -------------------------------
with tab_backtest:
    st.subheader("Simple 1-Day Hold Backtest (Test Segment)")
    if "y_pred_eval" not in st.session_state:
        st.warning("Evaluate the model first in the 'Evaluation' tab.")
    else:
        y_pred_eval = st.session_state["y_pred_eval"]
        sig = pd.Series(y_pred_eval, index=data.index[split_idx:]).astype(float)

        nxt_ret = data["Close"].pct_change().shift(-1)
        strat = (nxt_ret.loc[sig.index] * sig).fillna(0)

        cost = 0.0005
        open_trade = (sig.shift(1).fillna(0) < 1) & (sig == 1)
        strat[open_trade] -= cost

        eq = (1 + strat).cumprod()
        st.line_chart(eq, height=280)

        total_return = float(eq.iloc[-1] - 1.0)
        sharpe_like = float((strat.mean() / (strat.std() + 1e-12)) * np.sqrt(252))
        trades = int(open_trade.sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Return (Test)", f"{total_return:.2%}")
        c2.metric("Trades", f"{trades}")
        c3.metric("Sharpe-like", f"{sharpe_like:.2f}")

# -------------------------------
# Predict
# -------------------------------
with tab_predict:
    st.subheader("Next-Day Prediction (Final Result)")

    best_est = st.session_state["best_est"]
    th = st.session_state.get("cv_threshold", 0.5)

    ind_only = apply_feature_clipping(compute_indicators_only(df))
    missing = [c for c in feature_cols if c not in ind_only.columns]
    if missing:
        st.error(f"Missing features for prediction: {missing}")
    else:
        X_last = ind_only[feature_cols].replace([np.inf, -np.inf], np.nan).iloc[[-1]]

        if hasattr(best_est, "predict_proba"):
            proba_last = float(best_est.predict_proba(X_last)[:, 1][0])
        elif hasattr(best_est, "decision_function"):
            score = float(best_est.decision_function(X_last)[0])
            proba_last = 1 / (1 + np.exp(-score))
        else:
            proba_last = float(best_est.predict(X_last)[0])

        raw_label = int(proba_last >= th)

        final_label = raw_label
        trend_gate = "N/A"
        if use_trend_filter:
            close_s = ind_only["Close"]
            sma50 = close_s.rolling(50).mean()
            sma200 = close_s.rolling(200).mean()
            in_trend = bool((sma50.iloc[-1] > sma200.iloc[-1]))
            trend_gate = "Allowed (Uptrend)" if in_trend else "Blocked (Downtrend)"
            final_label = raw_label * int(in_trend)

        direction = "Up" if final_label == 1 else "Down"
        confidence = float(proba_last)

        if direction == "Up" and trend_gate.startswith("Allowed"):
            recommendation = "BUY"; explanation = "Model expects price to increase and market trend supports buying."
        elif direction == "Up" and "Blocked" in trend_gate:
            recommendation = "HOLD"; explanation = "Model expects upward move but market trend is weak. Wait for confirmation."
        elif direction == "Down" and trend_gate.startswith("Allowed"):
            recommendation = "HOLD"; explanation = "Model sees weakness but overall trend is still positive. Avoid selling prematurely."
        else:
            recommendation = "SELL"; explanation = "Model expects price to decrease and the trend is bearish."

        badge_color = {"BUY": "#16A34A", "HOLD": "#f59e0b", "SELL": "#DC2626"}[recommendation]
        st.markdown(
            f"""
            <div style="padding:10px 12px;border-radius:10px;background:{badge_color}22;
                        border:1px solid {badge_color};display:inline-block;">
                <span style="font-weight:700;color:{badge_color};">📌 Recommendation: {recommendation}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Direction (Tomorrow)", direction)
        c2.metric("Confidence (Up)", f"{confidence:.1%}")
        c3.metric("Market Trend", trend_gate)

        if abs(confidence - th) < 0.05:
            st.warning("Confidence is close to the decision threshold. Treat as low conviction.")

        st.info(explanation)
        st.caption("We predict whether tomorrow’s close will be higher than today’s. Threshold tuned via CV.")

# -------------------------------
# Insights
# -------------------------------
with tab_insights:
    st.subheader("Insights")

    best_est = st.session_state["best_est"]
    model_name = st.session_state.get("model_choice", "Model")
    th = float(st.session_state.get("cv_threshold", 0.5))

    best_est.fit(X_train, y_train)

    if hasattr(best_est, "predict_proba"):
        y_proba = best_est.predict_proba(X_test)[:, 1]
    elif hasattr(best_est, "decision_function"):
        s = best_est.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-s))
    else:
        y_proba = best_est.predict(X_test).astype(float)
    y_pred = (y_proba >= th).astype(int)

    # Importances (native or permutation)
    final_clf = best_est.named_steps.get("clf", best_est)
    importances = None
    try:
        if hasattr(final_clf, "feature_importances_"):
            vals = np.asarray(final_clf.feature_importances_, dtype=float)
            importances = pd.Series(vals, index=feature_cols)
        elif hasattr(final_clf, "coef_"):
            vals = np.abs(np.ravel(final_clf.coef_))
            importances = pd.Series(vals, index=feature_cols)
    except Exception:
        importances = None

    if importances is None:
        from sklearn.inspection import permutation_importance
        st.caption("Using permutation importance (model has no native importances).")
        n = min(len(X_test), 400)
        r = permutation_importance(best_est, X_test[:n], y_test[:n],
                                   n_repeats=8, random_state=GLOBAL_SEED, n_jobs=1)
        importances = pd.Series(r.importances_mean, index=feature_cols)

    topk = importances.sort_values(ascending=False).head(12)
    plot_df = pd.DataFrame({"Feature": topk.index, "Importance": topk.values}).sort_values("Importance")

    fig_imp = go.Figure(go.Bar(x=plot_df["Importance"], y=plot_df["Feature"], orientation="h"))
    fig_imp.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10),
                          title=f"What drives predictions — {model_name}",
                          xaxis_title="Relative importance", yaxis_title="Signal")
    st.plotly_chart(fig_imp, use_container_width=True)
