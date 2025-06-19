from data_loader import load_and_prepare_data
from model import train_and_predict
import matplotlib.pyplot as plt

if __name__ == "__main__":
    symbol = "TCS.NS"
    df, X_train, y_train, X_test, y_test, scaler = load_and_prepare_data(symbol)
    model, predictions = train_and_predict(X_train, y_train, X_test, y_test)

    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_rescaled = scaler.inverse_transform(predictions)

    plt.figure(figsize=(10, 5))
    plt.plot(y_test_rescaled, label='Actual')
    plt.plot(predictions_rescaled, label='Predicted')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_plot.png")
    plt.show()
