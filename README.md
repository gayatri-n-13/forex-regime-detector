# 📡 Forex Market Regime Intelligence

A real-time forex market state detection system using a **Gaussian Hidden Markov Model** 
on 11 engineered features, with live signal generation and strategy backtesting.

---

## What It Does

Automatically detects 4 market regimes in EUR/USD, GBP/USD, USD/JPY, AUD/USD:

| Regime | Meaning | Signal |
|---|---|---|
| 📈 Bull Trend | Low vol, positive drift | Long |
| 📉 Bear Trend | Low vol, negative drift | Short |
| ⚡ High Vol | Explosive movement, unclear direction | Flat |
| 〰️ Low Vol Range | Consolidation, no trend | Flat |

---

## Why These Technical Choices

**Gaussian HMM over K-Means:**
K-Means assigns regimes purely on distance — it has no memory.
Markets don't teleport between states; they transition probabilistically.
HMM models that transition structure explicitly via the transition matrix,
giving you both the current regime AND the probability of the next one.

**Full covariance (not diagonal):**
Diagonal covariance assumes features are independent — they're not.
Volatility and RSI move together. Full covariance captures those relationships.

**Garman-Klass volatility over close-to-close:**
Close-to-close ignores what happened *during* the bar.
GK uses OHLC and is 5x more statistically efficient — standard in microstructure research.

**Multi-horizon vol ratio (8h / 72h):**
A single volatility window misses regime transitions.
The ratio captures whether short-term vol is expanding or contracting
relative to the longer-term baseline — a leading signal of regime change.

**11 features instead of 4:**
Returns + RSI alone don't separate trending from ranging markets well.
Adding BB%B, ATR ratio, trend strength (rolling correlation proxy for ADX),
and momentum Z-score gives the HMM enough signal to find stable, 
interpretable hidden states.

---

## Architecture
