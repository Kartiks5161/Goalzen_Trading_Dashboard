# streamlit_app.py
import warnings
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, average_precision_score

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Stock Direction ML — Final", layout="wide")

# -------------------------------
# Helpers: loading & features
# -------------------------------
def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten columns and normalize to lowercase_with_underscores."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in t if str(x)]).strip() for t in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def load_ohlcv(ticker: str, years: int) -> pd.DataFrame:
    """Robust daily OHLCV loader that survives Yahoo quirks."""
    df = yf.download(
        tickers=ticker,
        period=f"{years}y",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df is None or len(df) == 0:
        raise RuntimeError("No data returned from Yahoo. Check ticker/internet.")
    df = _flatten_cols(df.dropna().copy().sort_index())

    # Pick first column containing each key
    def pick(key, prefer_not=None):
        cols = [c for c in df.columns if key in c]
        if prefer_not:
            cols = [c for c in cols if prefer_not not in c] or cols
        return cols[0] if cols else None

    open_c  = pick("open")
    high_c  = pick("high")
    low_c   = pick("low")
    close_c = pick("close", prefer_not="adj")
    vol_c   = pick("volume")

    if close_c is None or vol_c is None:
        preview = list(df.columns)[:10]
        raise RuntimeError(f"Could not find close/volume columns. Seen: {preview}")

    out = pd.DataFrame(index=df.index)
    if open_c and high_c and low_c:
        out["Open"] = pd.to_numeric(df[open_c], errors="coerce")
        out["High"] = pd.to_numeric(df[high_c], errors="coerce")
        out["Low"]  = pd.to_numeric(df[low_c],  errors="coerce")
    out["Close"]  = pd.to_numeric(df[close_c], errors="coerce")
    out["Volume"] = pd.to_numeric(df[vol_c],   errors="coerce")

    return out.dropna(subset=["Close", "Volume"]).copy()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Features for TRAINING + target y (next-day Up/Down)."""
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

    # Target (NEXT DAY up vs down)
    o["y"] = (close.shift(-1) > close).astype(int)

    return o.dropna().copy()

def compute_indicators_only(df: pd.DataFrame) -> pd.DataFrame:
    """Same indicators as training, but WITHOUT 'y' — for final prediction."""
    return compute_indicators(df).drop(columns=["y"])

# -------------------------------
# Modeling helpers
# -------------------------------
def build_models():
    preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
    ])

    models = {
        "LogisticRegression": Pipeline([
            ("prep", preprocess),
            ("scale", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=2000, random_state=42))
        ]),

        "RandomForest": Pipeline([
            ("prep", preprocess),
            ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))
        ]),

        "HistGradientBoosting": Pipeline([
            ("prep", preprocess),
            ("clf", HistGradientBoostingClassifier(random_state=42))
        ]),

        "XGBoost": Pipeline([
            ("prep", preprocess),
            ("clf", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_estimators=300,
                learning_rate=0.05
            ))
        ]),

        "CatBoost": Pipeline([
            ("prep", preprocess),
            ("clf", CatBoostClassifier(
                verbose=0,
                random_state=42,
                iterations=300,
                learning_rate=0.05,
                depth=6
            ))
        ]),
    }

    grids = {
        "LogisticRegression": {"clf__C": [0.2, 1.0, 3.0]},

        "RandomForest": {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [6, 10, None]
        },

        "HistGradientBoosting": {
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [None, 6, 10]
        },

        "XGBoost": {
            "clf__max_depth": [3, 5],
            "clf__subsample": [0.8, 1.0]
        },

        "CatBoost": {
            "clf__depth": [4, 6],
            "clf__l2_leaf_reg": [1, 5]
        },
    }

    return models, grids

def tune_threshold_cv(best_estimator, X_train, y_train, n_splits=5):
    """Choose probability threshold via time-aware CV on the training set."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    probs = np.zeros_like(y_train, dtype=float)
    for tr, va in tscv.split(X_train):
        m = best_estimator
        m.fit(X_train[tr], y_train[tr])
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(X_train[va])[:, 1]
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
# Sidebar
# -------------------------------
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker (Yahoo Finance)", value="RELIANCE.NS")
years = st.sidebar.slider("Years of history", 1, 5, 2)
use_trend_filter = st.sidebar.checkbox("Trend Filter (SMA50 > SMA200)", value=True)

# -------------------------------
# Tabs
# -------------------------------
tab_overview, tab_train, tab_eval, tab_backtest, tab_predict, tab_insights = st.tabs(
    ["Overview", "Train", "Evaluation", "Backtest", "Predict", "Insights"]
)


# -------------------------------
# Shared data & features
# -------------------------------
df = load_ohlcv(ticker, years)           # Open/High/Low/Close/Volume (if OHLC present)
feat = compute_indicators(df)            # indicators + y

# --- Minimal Robust Outlier Clipping (Option A) ---
# Cap daily returns to avoid shock gaps
if "Return" in feat.columns:
    feat["Return"] = feat["Return"].clip(lower=-0.15, upper=0.15)

# Cap rolling volatility (just in case)
if "Volatility_10" in feat.columns:
    feat["Volatility_10"] = feat["Volatility_10"].clip(upper=0.20)
if "Volatility_20" in feat.columns:
    feat["Volatility_20"] = feat["Volatility_20"].clip(upper=0.20)

# Smooth volume spikes
if "Volume_Change" in feat.columns:
    feat["Volume_Change"] = feat["Volume_Change"].clip(lower=-5, upper=5)
if "Vol_to_SMA20" in feat.columns:
    feat["Vol_to_SMA20"] = feat["Vol_to_SMA20"].clip(lower=0, upper=5)

# Remove any infinite values that might remain
feat = feat.replace([np.inf, -np.inf], np.nan).dropna().copy()


data = feat.dropna().copy()
feature_cols = [c for c in data.columns if c != "y"]

X = data[feature_cols].replace([np.inf, -np.inf], np.nan).values
y = data["y"].values

split_idx = int(0.8 * len(data))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# -------------------------------
# Overview
# -------------------------------
with tab_overview:
    # --- Time Range Selector ---
    range_choice = st.selectbox(
        "Show price history for:",
        ["30D", "90D", "180D", "365D", "All"],
        index=2  # default = 180D
    )

    def slice_by_range(df_in, choice):
        if choice == "All":
            return df_in
        days = int(choice[:-1])   # "180D" -> 180
        return df_in.tail(days)

    df_view = slice_by_range(df, range_choice)

    st.subheader("Price • Volume • RSI • MACD")

    have_ohlc = all(c in df_view.columns for c in ["Open", "High", "Low", "Close"])
    have_vol  = "Volume" in df_view.columns

    # Compute RSI(14) and MACD(12,26,9) from Close
    close = df_view["Close"].astype(float)
    # RSI
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi14 = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    # Layout: 4 rows -> Price, Volume, RSI, MACD
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        vertical_spacing=0.03,
        subplot_titles=("Price", "Volume", "RSI (14)", "MACD (12,26,9)")
    )

    # Row 1: Price
    if have_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=df_view.index,
                open=df_view["Open"].astype(float),
                high=df_view["High"].astype(float),
                low=df_view["Low"].astype(float),
                close=df_view["Close"].astype(float),
                increasing_line_color="#16A34A",
                decreasing_line_color="#DC2626",
                name="Price"
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(x=df_view.index, y=close, mode="lines", name="Close"),
            row=1, col=1
        )

    # Row 2: Volume (if available)
    if have_vol:
        vol_colors = np.where(
            df_view["Close"].values >= (df_view["Open"].values if have_ohlc else df_view["Close"].shift(1).values),
            "#16A34A", "#DC2626"
        )
        fig.add_trace(
            go.Bar(
                x=df_view.index,
                y=df_view["Volume"].astype(float),
                marker_color=vol_colors,
                opacity=0.7,
                name="Volume"
            ),
            row=2, col=1
        )

    # Row 3: RSI
    fig.add_trace(
        go.Scatter(x=df_view.index, y=rsi14, mode="lines", name="RSI(14)"),
        row=3, col=1
    )
    # RSI guides
    fig.add_hline(y=70, line_dash="dot", line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_width=1, row=3, col=1)

    # Row 4: MACD + signal + histogram
    fig.add_trace(
        go.Bar(x=df_view.index, y=macd_hist, name="MACD Hist", opacity=0.6),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_view.index, y=macd, mode="lines", name="MACD"),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_view.index, y=macd_signal, mode="lines", name="Signal"),
        row=4, col=1
    )

    fig.update_layout(
        height=900,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        showlegend=False
    )
    fig.update_yaxes(title_text=f"{ticker} Price", row=1, col=1)
    if have_vol:
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)


with tab_train:
    st.subheader("Model Training (Select algorithm + GridSearchCV)")

    # get all models + grids
    models, grids = build_models()
    model_names = list(models.keys())
    default_idx = model_names.index("HistGradientBoosting") if "HistGradientBoosting" in model_names else 0

    # pick one
    model_choice = st.selectbox("Choose model", model_names, index=default_idx)
    pipe = models[model_choice]
    grid = grids[model_choice]

    tscv = TimeSeriesSplit(n_splits=5)
    with st.spinner("Training & tuning..."):
        gs = GridSearchCV(pipe, grid, scoring="f1", cv=tscv, n_jobs=-1, refit=True, verbose=0)
        gs.fit(X_train, y_train)

    best_est = gs.best_estimator_
    st.success(f"{model_choice} — Best CV F1: {gs.best_score_:.4f}")
    st.caption("Best hyperparameters:")
    st.json(gs.best_params_)

    # threshold from CV on train only
    th_cv, thcv_f1 = tune_threshold_cv(best_est, X_train, y_train, n_splits=5)
    st.info(f"Auto-tuned threshold from CV: {th_cv:.3f}  (CV F1 ≈ {thcv_f1:.4f})")

    # stash in session
    st.session_state["best_est"] = best_est
    st.session_state["cv_threshold"] = th_cv
    st.session_state["best_params"] = gs.best_params_
    st.session_state["model_choice"] = model_choice

# -------------------------------
# Evaluation
# -------------------------------
with tab_eval:
    st.subheader("Holdout Evaluation")
    if "best_est" not in st.session_state:
        st.warning("Train the model in the 'Train' tab first.")
    else:
        best_est = st.session_state["best_est"]
        th = st.session_state["cv_threshold"]

        best_est.fit(X_train, y_train)
        y_proba = best_est.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= th).astype(int)

        # optional trend filter
        if use_trend_filter:
            close = data["Close"]
            sma50 = close.rolling(50).mean()
            sma200 = close.rolling(200).mean()
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
        model_name = st.session_state.get("model_choice", "Model")

        with col_a:
            st.pyplot(plot_cm_small(cm, f"Confusion Matrix — {model_name}"))

        with col_b:
            st.pyplot(plot_roc_small(y_test, y_proba, f"ROC Curve — {model_name}"))


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

        # Small entry cost
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
# Predict (final result)
# -------------------------------
with tab_predict:
    st.subheader("Next-Day Prediction (Final Result)")

    if "best_est" not in st.session_state:
        st.warning("Train the model in the 'Train' tab first.")
    else:
        best_est = st.session_state["best_est"]
        th = st.session_state.get("cv_threshold", 0.5)

        # Build latest feature row (no 'y')
        ind_only = compute_indicators_only(df)
        missing = [c for c in feature_cols if c not in ind_only.columns]
        if missing:
            st.error(f"Missing features for prediction: {missing}")
        else:
            X_last = ind_only[feature_cols].replace([np.inf, -np.inf], np.nan).iloc[[-1]]

            # Robust probability
            if hasattr(best_est, "predict_proba"):
                proba_last = float(best_est.predict_proba(X_last)[:, 1][0])
            elif hasattr(best_est, "decision_function"):
                score = float(best_est.decision_function(X_last)[0])
                # rank-based squashing to 0..1
                proba_last = 1 / (1 + np.exp(-score))
            else:
                proba_last = float(best_est.predict(X_last)[0])

            raw_label = int(proba_last >= th)

            # Optional trend gate
            final_label = raw_label
            trend_gate = "N/A"
            if use_trend_filter:
                close = ind_only["Close"]
                sma50 = close.rolling(50).mean()
                sma200 = close.rolling(200).mean()
                in_trend = bool((sma50.iloc[-1] > sma200.iloc[-1]))
                trend_gate = "Allowed (Uptrend)" if in_trend else "Blocked (Downtrend)"
                final_label = raw_label * int(in_trend)

            # Convert to user-friendly recommendation
            direction = "Up" if final_label == 1 else "Down"
            confidence = float(proba_last)

            if direction == "Up" and trend_gate.startswith("Allowed"):
                recommendation = "BUY"
                explanation = "Model expects price to increase and market trend supports buying."
            elif direction == "Up" and "Blocked" in trend_gate:
                recommendation = "HOLD"
                explanation = "Model expects upward move but market trend is weak. Wait for confirmation."
            elif direction == "Down" and trend_gate.startswith("Allowed"):
                recommendation = "HOLD"
                explanation = "Model sees weakness but overall trend is still positive. Avoid selling prematurely."
            else:  # Down + Blocked
                recommendation = "SELL"
                explanation = "Model expects price to decrease and the trend is bearish."

            # Pretty header badge
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

            # Caution if confidence is low relative to threshold
            if abs(confidence - th) < 0.05:
                st.warning(
                    "Confidence is close to the decision threshold. Treat as **low conviction**; "
                    "consider waiting for more confirmation."
                )

            st.info(explanation)
            st.caption(
                "We predict whether tomorrow’s close will be higher than today’s. "
                "Confidence is the model’s Up-probability; the decision threshold was tuned via CV."
            )


# -------------------------------
# Insights
# -------------------------------
# -------------------------------
# Insights
# -------------------------------
with tab_insights:
    st.subheader("Insights")

    if "best_est" not in st.session_state:
        st.warning("Train the model first in the 'Train' tab.")
    else:
        best_est = st.session_state["best_est"]
        model_name = st.session_state.get("model_choice", "Model")
        th = float(st.session_state.get("cv_threshold", 0.5))

        # Refit on train for stable importances & outputs
        best_est.fit(X_train, y_train)

        # Robust probabilities on test set
        if hasattr(best_est, "predict_proba"):
            y_proba = best_est.predict_proba(X_test)[:, 1]
        elif hasattr(best_est, "decision_function"):
            s = best_est.decision_function(X_test)
            y_proba = 1 / (1 + np.exp(-s))
        else:
            y_proba = best_est.predict(X_test).astype(float)

        y_pred = (y_proba >= th).astype(int)

        # =========================================================
        # 1) What drives the prediction (human-friendly importances)
        # =========================================================
        st.markdown("### 1) What drives the prediction")

        # Try native importances; fallback to permutation importance
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
            st.caption("Using permutation importance (model has no native importances).")
            n = min(len(X_test), 500)
            r = permutation_importance(
                best_est, X_test[:n], y_test[:n],
                n_repeats=10, random_state=42, n_jobs=-1
            )
            importances = pd.Series(r.importances_mean, index=feature_cols)

        # Friendly labels & categories
        friendly = {
            "Return": "Today’s % price move",
            "Return_lag_1": "Yesterday’s % move",
            "Return_lag_2": "2-day lag return",
            "Return_lag_3": "3-day lag return",
            "Volatility_10": "10-day volatility",
            "Volatility_20": "20-day volatility",
            "Volume": "Trading volume",
            "Volume_Change": "Change in volume",
            "Vol_to_SMA20": "Volume vs 20-day avg",
            "BB_Width": "Bollinger band width",
            "BB_Upper": "Bollinger upper band",
            "BB_Lower": "Bollinger lower band",
            "SMA_10": "10-day average",
            "SMA_20": "20-day average",
            "SMA_50": "50-day average",
            "SMA_100": "100-day average",
            "EMA_5": "5-day EMA",
            "EMA_12": "12-day EMA",
            "EMA_26": "26-day EMA",
            "Close_to_SMA_10": "Price vs 10-day avg",
            "Close_to_SMA_20": "Price vs 20-day avg",
            "RSI_14": "RSI (14)",
            "MACD": "MACD",
            "MACD_Signal": "MACD signal",
            "MACD_Hist": "MACD histogram",
        }

        def category(feat: str) -> str:
            f = feat.lower()
            if "volume" in f: return "Participation (Volume)"
            if "rsi" in f or "macd" in f: return "Momentum Oscillators"
            if "bb_" in f or "volatility" in f: return "Volatility"
            if "sma" in f or "ema" in f or "close_to" in f: return "Trend / Averages"
            if "return" in f: return "Short-term Momentum"
            return "Other"

        explain = {
            "Participation (Volume)": "Bigger volume/changes show stronger conviction.",
            "Momentum Oscillators":  "Momentum turns & overbought/oversold zones (RSI/MACD).",
            "Volatility":            "Higher volatility often precedes bigger swings.",
            "Trend / Averages":      "Price vs moving averages indicates trend direction.",
            "Short-term Momentum":   "Recent moves tend to carry into the next day.",
            "Other":                 "Minor supporting signals."
        }

        cat_colors = {
            "Participation (Volume)": "#3b82f6",
            "Momentum Oscillators":  "#a855f7",
            "Volatility":             "#ef4444",
            "Trend / Averages":       "#10b981",
            "Short-term Momentum":    "#f59e0b",
            "Other":                  "#6b7280",
        }

        topk = importances.sort_values(ascending=False).head(12)
        plot_df = pd.DataFrame({"Feature": topk.index, "Importance": topk.values})
        plot_df["Label"] = plot_df["Feature"].map(lambda x: friendly.get(x, x))
        plot_df["Category"] = plot_df["Feature"].map(category)
        plot_df["Color"] = plot_df["Category"].map(cat_colors)
        plot_df = plot_df.sort_values("Importance")  # bottom→top

        fig_imp = go.Figure(go.Bar(
            x=plot_df["Importance"],
            y=plot_df["Label"],
            orientation="h",
            marker=dict(color=plot_df["Color"]),
            customdata=np.stack([
                plot_df["Category"].values,
                plot_df["Category"].map(explain).values
            ], axis=1),
            hovertemplate="<b>%{y}</b><br>Group: %{customdata[0]}<br>"
                          "Why: %{customdata[1]}<br>Importance: %{x:.4f}<extra></extra>"
        ))
        fig_imp.update_layout(
            height=500, margin=dict(l=10, r=10, t=50, b=10),
            title=f"What drives predictions — {model_name}",
            xaxis_title="Relative importance (higher = stronger influence)",
            yaxis_title="Signal"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        # 3 quick takeaways
        cat_share = plot_df.groupby("Category")["Importance"].sum().sort_values(ascending=False)
        bullets = [f"- **{cat}** — {explain.get(cat, '')}" for cat in list(cat_share.index[:3])]
        st.markdown("**Takeaways:**")
        st.markdown("\n".join(bullets))
        st.caption("Higher bars = stronger influence on Up/Down decisions for this dataset.")

        # =========================================================
        # 2) Actual vs Predicted — Price + Outcome Timeline
        # =========================================================
        st.markdown("### 2) Actual vs Predicted")

        pred_df = pd.DataFrame({
            "date": data.index[split_idx:],
            "close": data["Close"].iloc[split_idx:].values,
            "y_true": y_test,
            "proba_up": y_proba,
            "y_pred": y_pred
        }).set_index("date")

        col1, col2 = st.columns(2)

        # Left: Price with BUY markers
        with col1:
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=pred_df.index, y=pred_df["close"],
                mode="lines+markers", name="Close", marker_size=4
            ))
            buy_idx = pred_df.index[pred_df["y_pred"] == 1]
            if len(buy_idx):
                fig_price.add_trace(go.Scatter(
                    x=buy_idx, y=pred_df.loc[buy_idx, "close"],
                    mode="markers", name="Predicted BUY",
                    marker_symbol="triangle-up", marker_size=10
                ))
            fig_price.update_layout(
                title=f"{ticker} — Price with Predicted BUYs",
                height=380, margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Date", yaxis_title=f"{ticker} Price",
                xaxis_rangeslider_visible=False, hovermode="x unified"
            )
            st.plotly_chart(fig_price, use_container_width=True)

        # Right: Outcome timeline (TP/FP/FN/TN) + mini hit-rate
        with col2:
            # Build categories
            cats = []
            for yt, yp in zip(pred_df["y_true"].values, pred_df["y_pred"].values):
                if yp == 1 and yt == 1:
                    cats.append("Correct BUY (TP)")
                elif yp == 1 and yt == 0:
                    cats.append("Wrong BUY (FP)")
                elif yp == 0 and yt == 1:
                    cats.append("Missed Up (FN)")
                else:
                    cats.append("Correct NO-BUY (TN)")

            color_map = {
                "Correct BUY (TP)": "#16A34A",
                "Wrong BUY (FP)":   "#DC2626",
                "Missed Up (FN)":   "#F59E0B",
                "Correct NO-BUY (TN)": "#6B7280"
            }
            colors = [color_map[c] for c in cats]

            fig_out = go.Figure(go.Bar(
                x=pred_df.index, y=[1]*len(pred_df),
                marker_color=colors, marker_line_width=0,
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{customdata}<extra></extra>",
                customdata=cats, name="Outcome"
            ))
            # Legend swatches under plot
            for name, colr in color_map.items():
                fig_out.add_trace(go.Bar(x=[None], y=[None], name=name, marker_color=colr))

            fig_out.update_layout(
                title="Prediction Outcomes",
                height=360, margin=dict(l=10, r=10, t=40, b=0),
                xaxis_title="Date", yaxis=dict(visible=False, range=[0, 1.05]),
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0)
            )
            st.plotly_chart(fig_out, use_container_width=True)

            # Mini cumulative accuracy
            hits = (pred_df["y_true"].values == pred_df["y_pred"].values).astype(int)
            cum_hit = np.cumsum(hits) / np.arange(1, len(hits)+1)
            fig_hr = go.Figure(go.Scatter(x=pred_df.index, y=cum_hit,
                                          mode="lines", name="Hit-rate"))
            fig_hr.update_layout(
                height=200, margin=dict(l=10, r=10, t=10, b=10),
                title="Cumulative Accuracy", yaxis_title="Accuracy", xaxis_title=None
            )
            st.plotly_chart(fig_hr, use_container_width=True)

        # =========================================================
        # 3) Model performance (KPIs + simple bars)
        # =========================================================
        st.markdown("### 3) Model Performance")

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        ap = average_precision_score(y_test, y_proba)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Overall Accuracy", f"{acc:.1%}")
        k2.metric("Precision (BUY quality)", f"{prec:.1%}")
        k3.metric("Recall (UP coverage)", f"{rec:.1%}")
        k4.metric("F1 Score", f"{f1:.2f}")
        k5.metric("Avg Precision (PR AUC)", f"{ap:.2f}")

        st.caption("How clean the BUYs are vs how many UP days we catch at the chosen threshold.")
        fig_prbars = go.Figure()
        fig_prbars.add_trace(go.Bar(name="Precision", x=["Now"], y=[prec]))
        fig_prbars.add_trace(go.Bar(name="Recall",    x=["Now"], y=[rec]))
        fig_prbars.update_layout(
            barmode="group", height=220,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[0, 1], tickformat=".0%"),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0)
        )
        st.plotly_chart(fig_prbars, use_container_width=True)

        # =========================================================
        # 4) Strategy vs Buy & Hold (robust & compact)
        # =========================================================
        st.markdown("### 4) Strategy vs Buy & Hold (Test Segment)")

        # Buy & Hold: use simple daily returns; normalize from first valid
        ret = data["Close"].pct_change()
        bh_series = (1 + ret.iloc[split_idx:]).dropna().cumprod()
        if len(bh_series):
            bh_series = bh_series / bh_series.iloc[0]

        # Strategy equity from next-day returns and predictions
        nxt_ret = data["Close"].pct_change().shift(-1)
        sig_series = pd.Series(y_pred, index=data.index[split_idx:]).astype(float)
        strat = (nxt_ret.loc[sig_series.index] * sig_series).fillna(0)

        # Entry cost on opening a long
        cost = 0.0005
        open_trade = (sig_series.shift(1).fillna(0) < 1) & (sig_series == 1)
        strat[open_trade] -= cost
        eq = (1 + strat).cumprod()

        # Align both series for a clean chart
        comp = pd.concat(
            [eq.rename("Strategy"), bh_series.rename("Buy & Hold")],
            axis=1
        ).dropna()

        st.line_chart(comp, height=260)

        final_strat = float(comp["Strategy"].iloc[-1] - 1.0) if len(comp) else 0.0
        final_bh    = float(comp["Buy & Hold"].iloc[-1] - 1.0) if len(comp) else 0.0

        m1, m2 = st.columns(2)
        m1.metric("Strategy Total Return", f"{final_strat:.2%}")
        m2.metric("Buy & Hold Total Return", f"{final_bh:.2%}")

        # Plain-language summary
        pred_buys = int((y_pred == 1).sum())
        tp = int(((y_pred == 1) & (y_test == 1)).sum())
        fn = int(((y_pred == 0) & (y_test == 1)).sum())
        win_rate_buys = (tp / pred_buys) if pred_buys > 0 else 0.0

        st.markdown("**What this means:**")
        st.markdown(
            f"- The model issued **{pred_buys} BUY signals**, of which **{tp}** were correct.  \n"
            f"- Win-rate on BUYs: **{win_rate_buys:.0%}**. Missed UP days: **{fn}**.  \n"
            f"- Over this test window, the strategy returned **{final_strat:.1%}** vs Buy & Hold **{final_bh:.1%}**."
        )
