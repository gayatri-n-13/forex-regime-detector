import yfinance as yf
import pandas as pd
import streamlit as st

PAIRS = {
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD",
}

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(pair: str = "EURUSD=X", period: str = "1y", interval: str = "1h") -> pd.DataFrame:
    """
    Download OHLCV data for a given forex pair.
    Returns a clean DataFrame with a flat column index.
    """
    raw = yf.download(pair, period=period, interval=interval, auto_adjust=True)

    if raw.empty:
        raise ValueError(f"No data returned for {pair}. Check ticker or network.")

    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    raw.index = pd.to_datetime(raw.index, utc=True)

    # Drop rows where market was closed (zero volume AND no price movement)
    raw = raw[~((raw["Volume"] == 0) & (raw["High"] == raw["Low"]))]
    raw = raw.dropna()

    return raw


@st.cache_data(ttl=3600, show_spinner=False)
def load_multiple_pairs(period: str = "1y", interval: str = "1h") -> dict[str, pd.DataFrame]:
    """Load all tracked pairs for correlation analysis."""
    result = {}
    for ticker, name in PAIRS.items():
        try:
            result[name] = load_data(ticker, period=period, interval=interval)
        except Exception:
            pass
    return result
