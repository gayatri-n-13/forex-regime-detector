import yfinance as yf

def load_data():
    data = yf.download(
        "EURUSD=X",
        period="1y",
        interval="1h"
    )

    return data