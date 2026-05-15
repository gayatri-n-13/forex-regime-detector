import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from feature_engineering import FEATURE_COLS

# ── Regime taxonomy ───────────────────────────────────────────────────────────
# These are auto-assigned after fitting based on each regime's return/vol profile
REGIME_LABELS = {
    "Bull Trend":    {"color": "#00C896", "icon": "📈"},
    "Bear Trend":    {"color": "#FF4B6E", "icon": "📉"},
    "High Vol":      {"color": "#FFB347", "icon": "⚡"},
    "Low Vol Range": {"color": "#7EB8F7", "icon": "〰️"},
}

N_REGIMES = 4


def _label_regimes(data: pd.DataFrame, model: GaussianHMM, scaler: StandardScaler) -> dict[int, str]:
    """
    Auto-label each hidden state by its mean return and volatility.
    Uses a 2x2 grid: high/low return × high/low volatility.
    """
    regime_stats = (
        data.groupby("Regime")[["log_returns", "vol_24h"]]
            .mean()
    )
    med_ret = regime_stats["log_returns"].median()
    med_vol = regime_stats["vol_24h"].median()

    label_map = {}
    for state, row in regime_stats.iterrows():
        high_ret = row["log_returns"] >= med_ret
        high_vol = row["vol_24h"] >= med_vol
        if high_ret and not high_vol:
            label_map[state] = "Bull Trend"
        elif not high_ret and not high_vol:
            label_map[state] = "Bear Trend"
        elif high_vol:
            label_map[state] = "High Vol"
        else:
            label_map[state] = "Low Vol Range"

    # Deduplicate: if two states map to same label, append index
    seen = {}
    for k, v in label_map.items():
        if v in seen.values():
            label_map[k] = v + f" {k}"
        seen[k] = label_map[k]

    return label_map


def detect_regimes(data: pd.DataFrame) -> tuple[pd.DataFrame, GaussianHMM, dict, np.ndarray]:
    """
    Fit a Gaussian HMM and annotate the dataframe with:
    - Regime        : integer state (0–3)
    - RegimeLabel   : named regime string
    - RegimeColor   : hex color for plotting
    - Confidence    : posterior probability of the most likely state
    - Signal        : trading signal derived from regime (+1 long, -1 short, 0 flat)
    - ForwardReturn : actual next-bar return (for backtest)

    Returns (data, model, label_map, transition_matrix)
    """
    features = data[FEATURE_COLS].copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    model = GaussianHMM(
        n_components=N_REGIMES,
        covariance_type="full",   # full covariance captures feature interactions
        n_iter=2000,
        tol=1e-5,
        random_state=42,
    )
    model.fit(X)

    hidden_states = model.predict(X)
    posteriors    = model.predict_proba(X)           # shape (T, n_regimes)
    confidence    = posteriors.max(axis=1)           # max posterior per bar

    data = data.copy()
    data["Regime"]     = hidden_states
    data["Confidence"] = confidence

    # ── Label regimes ─────────────────────────────────────────────────────────
    label_map = _label_regimes(data, model, scaler)
    data["RegimeLabel"] = data["Regime"].map(label_map)
    data["RegimeColor"] = data["RegimeLabel"].map(
        lambda l: REGIME_LABELS.get(l.split(" ")[0] + (" " + l.split(" ")[1] if len(l.split(" ")) > 1 else ""),
                  REGIME_LABELS.get(l, {"color": "#AAAAAA"}))["color"]
    )

    # ── Signals ───────────────────────────────────────────────────────────────
    # Bull Trend → long (+1), Bear Trend → short (-1), else flat (0)
    def _regime_to_signal(label: str) -> int:
        if "Bull" in label:   return  1
        if "Bear" in label:   return -1
        return 0

    data["Signal"] = data["RegimeLabel"].map(_regime_to_signal)

    # ── Forward return (strategy P&L input) ───────────────────────────────────
    data["ForwardReturn"] = data["log_returns"].shift(-1)

    # ── Transition matrix (row = from, col = to) ──────────────────────────────
    trans_mat = model.transmat_

    return data, model, label_map, trans_mat


def compute_strategy_performance(data: pd.DataFrame) -> pd.DataFrame:
    """
    Backtest the regime-based signal against buy-and-hold.
    Returns a DataFrame of cumulative returns for comparison.
    """
    df = data.dropna(subset=["ForwardReturn", "Signal"]).copy()

    df["StrategyReturn"]   = df["Signal"] * df["ForwardReturn"]
    df["CumStrategy"]      = df["StrategyReturn"].cumsum().apply(np.exp) - 1
    df["CumBuyAndHold"]    = df["log_returns"].cumsum().apply(np.exp) - 1

    # Sharpe ratio (annualised, 252 trading days × 24 hrs)
    n_periods_per_year = 252 * 24
    mean_r  = df["StrategyReturn"].mean()
    std_r   = df["StrategyReturn"].std()
    sharpe  = (mean_r / (std_r + 1e-10)) * np.sqrt(n_periods_per_year)

    # Max drawdown
    cum     = df["CumStrategy"]
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / (roll_max + 1)
    max_dd   = drawdown.min()

    df.attrs["sharpe"]  = round(sharpe, 2)
    df.attrs["max_dd"]  = round(max_dd * 100, 2)
    df.attrs["win_rate"] = round((df["StrategyReturn"] > 0).mean() * 100, 1)

    return df


def regime_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Compute per-regime summary statistics."""
    stats = (
        data.groupby("RegimeLabel")
            .agg(
                Count         = ("log_returns", "count"),
                AvgReturn     = ("log_returns", "mean"),
                Volatility    = ("vol_24h", "mean"),
                AvgConfidence = ("Confidence", "mean"),
                WinRate       = ("log_returns", lambda x: (x > 0).mean()),
            )
            .reset_index()
    )
    stats["AvgReturn"]     = (stats["AvgReturn"] * 10_000).round(2)   # in pips-equivalent (×10⁴)
    stats["Volatility"]    = (stats["Volatility"] * 100).round(4)
    stats["AvgConfidence"] = (stats["AvgConfidence"] * 100).round(1)
    stats["WinRate"]       = (stats["WinRate"] * 100).round(1)
    stats["Freq%"]         = (stats["Count"] / stats["Count"].sum() * 100).round(1)
    return stats