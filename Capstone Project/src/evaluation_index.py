import numpy as np

def rmsse(y_true, y_pred, y_train):
    """Root Mean Squared Scaled Error (M5 metric).
    Scales test error by the naive (lag-1) forecast error on the training
    series, making errors comparable across series of different scales and
    robust to intermittency. Lower is better; <1 beats the naive forecast.

    y_true, y_pred: test-period actuals & predictions (length h)
    y_train: the series' training-period actuals (for the scaling denominator)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    # denominator: MSE of naive (yesterday=today) forecast over training period
    naive_mse = np.mean(np.diff(y_train) ** 2)
    if naive_mse == 0:  # flat training series — avoid div by zero
        return np.nan

    forecast_mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(forecast_mse / naive_mse)


def mae(y_true, y_pred):
    """Mean Absolute Error — intuitive units (avg units off per day)."""
    return np.mean(np.abs(np.asarray(y_true, dtype=float) -
                          np.asarray(y_pred, dtype=float)))