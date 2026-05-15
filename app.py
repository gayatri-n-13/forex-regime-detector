import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from data_loader import load_data, load_multiple_pairs, PAIRS
from feature_engineering import create_features
from model import detect_regimes, compute_strategy_performance, regime_statistics, REGIME_LABELS, N_REGIMES

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forex Regime Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"]   { font-size: 1.6rem; font-weight: 700; }
  [data-testid="stMetricLabel"]   { font-size: 0.78rem; color: #888; }
  .regime-badge {
      display: inline-block;
      padding: 2px 10px;
      border-radius: 20px;
      font-size: 0.8rem;
      font-weight: 600;
      color: #fff;
  }
  .section-title { font-size: 1.1rem; font-weight: 600; margin: 1.2rem 0 0.4rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Regime Intelligence")
    st.caption("HMM-based market state detection")

    selected_pair_name = st.selectbox("Currency Pair", list(PAIRS.values()), index=0)
    selected_ticker    = {v: k for k, v in PAIRS.items()}[selected_pair_name]

    period   = st.selectbox("Lookback Period", ["3mo", "6mo", "1y", "2y"], index=2)
    interval = st.selectbox("Bar Interval",    ["1h", "4h", "1d"], index=0)

    n_regimes_override = st.slider("Number of Regimes", 2, 6, N_REGIMES)
    confidence_thresh  = st.slider("Min Confidence Filter (%)", 0, 100, 60) / 100

    st.divider()
    st.caption("Model: Gaussian HMM · Full covariance · 2000 iterations")
    st.caption("Features: 11 engineered signals incl. Garman-Klass vol, ADX proxy, BB%B")
    run = st.button("🔄 Re-run Detection", use_container_width=True)

# ── Load & process ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def full_pipeline(ticker, period, interval):
    raw  = load_data(ticker, period, interval)
    feat = create_features(raw)
    data, model, label_map, trans_mat = detect_regimes(feat)
    perf = compute_strategy_performance(data)
    stats = regime_statistics(data)
    return data, model, label_map, trans_mat, perf, stats

if run:
    st.cache_data.clear()

with st.spinner("Running HMM regime detection…"):
    try:
        data, model, label_map, trans_mat, perf, stats = full_pipeline(
            selected_ticker, period, interval
        )
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

# Current regime
latest        = data.iloc[-1]
current_label = latest["RegimeLabel"]
current_conf  = latest["Confidence"]
current_sig   = latest["Signal"]
regime_color  = REGIME_LABELS.get(current_label, {}).get("color", "#888")
regime_icon   = REGIME_LABELS.get(current_label, {}).get("icon", "❓")

signal_map = {1: ("LONG  ▲", "#00C896"), -1: ("SHORT ▼", "#FF4B6E"), 0: ("FLAT  —", "#888888")}
sig_text, sig_color = signal_map[current_sig]

# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"Forex Market Regime Intelligence — {selected_pair_name}")
st.caption(f"Gaussian Hidden Markov Model · {N_REGIMES} hidden states · {len(data):,} bars analysed")

# ── Top KPI row ───────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric("Current Regime",   f"{regime_icon} {current_label}")
k2.metric("Detection Conf.",  f"{current_conf*100:.1f}%")
k3.metric("Active Signal",    sig_text)
k4.metric("Strategy Sharpe",  f"{perf.attrs['sharpe']:.2f}")
k5.metric("Max Drawdown",     f"{perf.attrs['max_dd']:.1f}%")
k6.metric("Win Rate",         f"{perf.attrs['win_rate']:.1f}%")

st.divider()

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Regime Map",
    "📈 Strategy Backtest",
    "🔄 Transition Matrix",
    "🔬 Regime Statistics",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Regime Map
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_chart, col_dist = st.columns([3, 1])

    with col_chart:
        st.markdown('<div class="section-title">Price Series Coloured by Regime</div>', unsafe_allow_html=True)

        # Filter by confidence threshold
        plot_data = data.copy()
        plot_data.loc[plot_data["Confidence"] < confidence_thresh, "RegimeLabel"] = "Low Confidence"

        # Build scatter trace per regime for legend control
        fig_regime = go.Figure()

        for label in plot_data["RegimeLabel"].unique():
            mask = plot_data["RegimeLabel"] == label
            color = REGIME_LABELS.get(label, {}).get("color", "#999999")
            sub   = plot_data[mask]
            fig_regime.add_trace(go.Scatter(
                x=sub.index,
                y=sub["Close"],
                mode="markers",
                marker=dict(size=3, color=color, opacity=0.85),
                name=label,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"Regime: {label}<br>"
                    "Price: %{y:.5f}<br>"
                    "<extra></extra>"
                ),
            ))

        fig_regime.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=10, b=10),
            legend=dict(orientation="h", y=-0.15),
            xaxis_title=None,
            yaxis_title="Price",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_regime, use_container_width=True)

        # Confidence ribbon
        st.markdown('<div class="section-title">Detection Confidence Over Time</div>', unsafe_allow_html=True)
        fig_conf = go.Figure(go.Scatter(
            x=data.index,
            y=data["Confidence"] * 100,
            fill="tozeroy",
            line=dict(color="#7EB8F7", width=1),
            fillcolor="rgba(126,184,247,0.25)",
        ))
        fig_conf.add_hline(y=confidence_thresh * 100, line_dash="dash",
                           line_color="#FFB347", annotation_text="Threshold")
        fig_conf.update_layout(
            height=160,
            margin=dict(l=0, r=0, t=5, b=5),
            yaxis_title="Confidence %",
            xaxis_title=None,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_conf, use_container_width=True)

    with col_dist:
        st.markdown('<div class="section-title">Regime Distribution</div>', unsafe_allow_html=True)
        regime_counts = data["RegimeLabel"].value_counts().reset_index()
        regime_counts.columns = ["Regime", "Count"]
        fig_pie = px.pie(
            regime_counts,
            names="Regime",
            values="Count",
            color="Regime",
            color_discrete_map={
                r: REGIME_LABELS.get(r, {}).get("color", "#888") for r in regime_counts["Regime"]
            },
            hole=0.55,
        )
        fig_pie.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=10),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_pie.update_traces(textposition="outside", textinfo="label+percent")
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('<div class="section-title">Return Distribution by Regime</div>', unsafe_allow_html=True)
        fig_box = px.box(
            data,
            x="RegimeLabel",
            y="log_returns",
            color="RegimeLabel",
            color_discrete_map={
                r: REGIME_LABELS.get(r, {}).get("color", "#888") for r in data["RegimeLabel"].unique()
            },
            points=False,
        )
        fig_box.update_layout(
            height=300,
            showlegend=False,
            margin=dict(l=0, r=0, t=5, b=5),
            xaxis_title=None,
            yaxis_title="Log Returns",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_box, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Strategy Backtest
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    **Strategy logic:** Go **long** in *Bull Trend* regimes, **short** in *Bear Trend* regimes,
    **flat** (cash) during *High Vol* and *Low Vol Range* regimes. Signals filtered by HMM posterior
    confidence.
    """)

    c1, c2 = st.columns([2, 1])

    with c1:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=perf.index, y=perf["CumBuyAndHold"] * 100,
            name="Buy & Hold", line=dict(color="#888888", dash="dot"),
        ))
        fig_bt.add_trace(go.Scatter(
            x=perf.index, y=perf["CumStrategy"] * 100,
            name="HMM Regime Strategy", line=dict(color="#00C896", width=2),
            fill="tozeroy", fillcolor="rgba(0,200,150,0.08)",
        ))

        # Shade regime transitions
        prev_signal = None
        start_idx   = None
        for i, (idx, row) in enumerate(perf.iterrows()):
            sig = row["Signal"]
            if sig != prev_signal:
                if prev_signal is not None and start_idx is not None:
                    bg_color = {1: "rgba(0,200,150,0.07)", -1: "rgba(255,75,110,0.07)"}.get(prev_signal)
                    if bg_color:
                        fig_bt.add_vrect(x0=start_idx, x1=idx, fillcolor=bg_color, line_width=0)
                start_idx = idx
            prev_signal = sig

        fig_bt.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=10, b=10),
            yaxis_title="Cumulative Return (%)",
            legend=dict(orientation="h", y=-0.15),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        # Drawdown chart
        roll_max = perf["CumStrategy"].cummax()
        drawdown = ((perf["CumStrategy"] - roll_max) / (roll_max.abs() + 1e-9)) * 100
        fig_dd = go.Figure(go.Scatter(
            x=perf.index, y=drawdown,
            fill="tozeroy", line=dict(color="#FF4B6E", width=1),
            fillcolor="rgba(255,75,110,0.2)", name="Drawdown %",
        ))
        fig_dd.update_layout(
            height=160, margin=dict(l=0, r=0, t=5, b=5),
            yaxis_title="Drawdown %",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    with c2:
        st.markdown("#### Performance Summary")
        final_strat = perf["CumStrategy"].iloc[-1] * 100
        final_bah   = perf["CumBuyAndHold"].iloc[-1] * 100
        alpha       = final_strat - final_bah

        st.metric("Total Return (Strategy)",   f"{final_strat:+.2f}%")
        st.metric("Total Return (Buy & Hold)", f"{final_bah:+.2f}%")
        st.metric("Alpha vs. Buy & Hold",      f"{alpha:+.2f}%",
                  delta=f"{'▲' if alpha > 0 else '▼'} {abs(alpha):.2f}%")
        st.metric("Sharpe Ratio",  perf.attrs["sharpe"])
        st.metric("Max Drawdown",  f"{perf.attrs['max_dd']}%")
        st.metric("Win Rate",      f"{perf.attrs['win_rate']}%")

        # Monthly return heatmap data
        st.markdown("#### Regime Signal Timeline")
        sig_counts = perf["Signal"].value_counts()
        labels = {1: "Long", -1: "Short", 0: "Flat"}
        for s, count in sig_counts.items():
            pct = count / len(perf) * 100
            col = {1: "#00C896", -1: "#FF4B6E", 0: "#888"}[s]
            st.markdown(
                f'<span class="regime-badge" style="background:{col}">'
                f'{labels[s]}</span> {pct:.1f}% of bars ({count:,})',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Transition Matrix
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    The **regime transition matrix** shows the probability of moving from one market state to
    another on the next bar. High diagonal values indicate **persistent** (sticky) regimes.
    Off-diagonal mass reveals which regimes are unstable or likely to flip.
    """)

    c1, c2 = st.columns([1, 1])

    with c1:
        labels_ordered = [label_map.get(i, str(i)) for i in range(N_REGIMES)]
        pct_matrix = (trans_mat * 100).round(1)

        fig_trans = go.Figure(go.Heatmap(
            z=pct_matrix,
            x=labels_ordered,
            y=labels_ordered,
            colorscale="Blues",
            text=[[f"{v:.1f}%" for v in row] for row in pct_matrix],
            texttemplate="%{text}",
            textfont={"size": 13},
            showscale=True,
            zmin=0, zmax=100,
        ))
        fig_trans.update_layout(
            title="Regime Transition Probabilities (%)",
            xaxis_title="To Regime →",
            yaxis_title="From Regime ↓",
            height=400,
            margin=dict(l=0, r=0, t=40, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_trans, use_container_width=True)

    with c2:
        st.markdown("#### Regime Persistence (Stickiness)")
        st.caption("Probability of staying in the same regime next bar.")
        for i in range(N_REGIMES):
            lbl   = label_map.get(i, str(i))
            prob  = trans_mat[i, i] * 100
            color = REGIME_LABELS.get(lbl, {}).get("color", "#888")
            st.markdown(f"**{lbl}**")
            st.progress(int(prob), text=f"{prob:.1f}%")

        st.divider()
        st.markdown("#### Next-Bar Regime Forecast")
        st.caption("From current regime, probable next states:")

        current_state = int(latest["Regime"])
        next_probs    = trans_mat[current_state]
        for i, prob in enumerate(next_probs):
            lbl = label_map.get(i, str(i))
            if prob > 0.05:
                st.markdown(f"→ **{lbl}**: `{prob*100:.1f}%`")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Regime Statistics
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### Per-Regime Statistical Summary")
    st.caption("AvgReturn is in ×10⁴ (pips-equivalent). Volatility is annualised log-return std.")

    # Style the stats table
    def color_return(val):
        if val > 0:   return "color: #00C896; font-weight: 600"
        if val < 0:   return "color: #FF4B6E; font-weight: 600"
        return ""

    styled = (
        stats.style
             .map(color_return, subset=["AvgReturn"])
             .format({
                 "AvgReturn":     "{:+.2f}",
                 "Volatility":    "{:.4f}",
                 "AvgConfidence": "{:.1f}%",
                 "WinRate":       "{:.1f}%",
                 "Freq%":         "{:.1f}%",
             })
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Volatility Profile by Regime")

    fig_vol = px.bar(
        stats,
        x="RegimeLabel",
        y="Volatility",
        color="RegimeLabel",
        color_discrete_map={
            r: REGIME_LABELS.get(r, {}).get("color", "#888") for r in stats["RegimeLabel"]
        },
        text="Volatility",
    )
    fig_vol.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_vol.update_layout(
        showlegend=False, height=300,
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis_title=None,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    st.divider()
    st.markdown("#### Feature Importance Heuristic")
    st.caption("Mean absolute z-score per feature across all regimes — higher = more discriminating.")

    from feature_engineering import FEATURE_COLS
    from sklearn.preprocessing import StandardScaler

    feat_data = data[FEATURE_COLS].dropna()
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(feat_data)
    feat_df   = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
    feat_df["Regime"] = data.loc[feat_data.index, "RegimeLabel"].values

    importance = (
        feat_df.groupby("Regime")[FEATURE_COLS]
               .mean()
               .abs()
               .mean()
               .sort_values(ascending=False)
               .reset_index()
    )
    importance.columns = ["Feature", "MeanAbsZ"]

    fig_imp = px.bar(
        importance, x="MeanAbsZ", y="Feature", orientation="h",
        color="MeanAbsZ", color_continuous_scale="Blues",
    )
    fig_imp.update_layout(
        height=340, showlegend=False,
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis_title="Mean |Z-Score| across regimes",
        yaxis_title=None,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_imp, use_container_width=True)