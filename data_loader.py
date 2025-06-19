import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from indicators import add_indicators

def load_and_prepare_data(symbol):
    df = yf.download(symbol, period="1y", interval="1d")
    df.dropna(inplace=True)
    df = add_indicators(df)
    df.dropna(inplace=True)

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['Close']])
    
    seq_len = 50
    X, y = [], []
    for i in range(seq_len, len(df_scaled)):
        X.append(df_scaled[i - seq_len:i])
        y.append(df_scaled[i])
    X, y = np.array(X), np.array(y)

    split = int(0.8 * len(X))
    return df, X[:split], y[:split], X[split:], y[split:], scaler
