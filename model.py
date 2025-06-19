from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_and_predict(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    predictions = model.predict(X_test)
    return model, predictions
