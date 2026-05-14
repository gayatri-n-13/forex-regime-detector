import ta

def create_features(data):

    # Convert Close column to 1D series
    close = data['Close'].squeeze()

    # Returns
    data['returns'] = close.pct_change()

    # Volatility
    data['volatility'] = (
        data['returns']
        .rolling(24)
        .std()
    )

    # RSI
    data['rsi'] = ta.momentum.RSIIndicator(
        close
    ).rsi()

    # MACD
    macd = ta.trend.MACD(close)

    data['macd'] = macd.macd()

    data = data.dropna()

    return data