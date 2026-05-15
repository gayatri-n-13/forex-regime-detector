import pandas as pd
import numpy as np
import ta


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build a rich feature set for regime detection.

    Features:
    - Returns & multi-horizon volatility
    - RSI, MACD, Bollinger %B, ATR ratio
    - Garman-Klass volatility estimator (microstructure)
    - Trend strength (ADX proxy via rolling correlation)
    - Volume-weighted price deviation
    - Momentum z-score
    """
    df = data.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]

    # ── Returns ───────────────────────────────────────────────────────────────
    df["returns"] = close.pct_change()
    df["log_returns"] = np.log(close / close.shift(1))

    # ── Volatility: rolling std at multiple windows ───────────────────────────
    df["vol_8h"]  = df["log_returns"].rolling(8).std()
    df["vol_24h"] = df["log_returns"].rolling(24).std()
    df["vol_72h"] = df["log_returns"].rolling(72).std()

    # Vol ratio: short-term vs long-term (regime expansion/contraction signal)
    df["vol_ratio"] = df["vol_8h"] / (df["vol_72h"] + 1e-10)

    # ── Garman-Klass Volatility (microstructure-aware) ────────────────────────
    # Uses OHLC — more efficient than close-to-close
    log_hl = (np.log(high / low)) ** 2
    log_co = (np.log(close / open_)) ** 2
    df["gk_vol"] = np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co)

    # ── Momentum Indicators ───────────────────────────────────────────────────
    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd_ind = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"]   = macd_ind.macd_diff()

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_pct"]   = bb.bollinger_pband()   # 0=lower band, 1=upper band
    df["bb_width"] = bb.bollinger_wband()   # band width (volatility proxy)

    # ── ATR Ratio (normalised range) ──────────────────────────────────────────
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["atr_ratio"] = atr / (close + 1e-10)

    # ── Trend Strength: rolling correlation of close with time index ──────────
    # Strong positive → uptrend; strong negative → downtrend; near 0 → ranging
    idx_num = pd.Series(range(len(df)), index=df.index)
    df["trend_strength"] = (
        close.rolling(24)
             .corr(idx_num)
    )

    # ── Momentum Z-Score ──────────────────────────────────────────────────────
    mom = close.pct_change(12)  # 12-hour momentum
    df["momentum_z"] = (mom - mom.rolling(72).mean()) / (mom.rolling(72).std() + 1e-10)

    df = df.dropna()
    return df


FEATURE_COLS = [
    "log_returns",
    "vol_24h",
    "vol_ratio",
    "gk_vol",
    "rsi",
    "macd_hist",
    "bb_pct",
    "bb_width",
    "atr_ratio",
    "trend_strength",
    "momentum_z",
]
