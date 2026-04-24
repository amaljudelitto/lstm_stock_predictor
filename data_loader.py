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

    feature_columns = ['Close', 'EMA20', 'RSI', 'MACD']
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
   df_features_scaled = feature_scaler.fit_transform(df[feature_columns])
    df_target_scaled = target_scaler.fit_transform(df[['Close']])
    
    seq_len = 50
    X, y = [], []
    for i in range(seq_len, len(df_features_scaled)):
        X.append(df_features_scaled[i - seq_len:i])
        y.append(df_target_scaled[i][0])
    X, y = np.array(X), np.array(y)

    split = int(0.8 * len(X))
    return df, X[:split], y[:split], X[split:], y[split:], target_scaler
