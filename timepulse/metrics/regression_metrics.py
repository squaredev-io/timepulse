import tensorflow as tf
import numpy as np


def calculate_mae(y_true: np.array, y_pred: np.array) -> float:
    """Calculate Mean Absolute Error."""
    return tf.keras.metrics.mean_absolute_error(y_true, y_pred).numpy()


def calculate_mse(y_true: np.array, y_pred: np.array) -> float:
    """Calculate Mean Squared Error."""
    return tf.keras.metrics.mean_squared_error(y_true, y_pred).numpy()


def calculate_rmse(y_true: np.array, y_pred: np.array) -> float:
    """Calculate Root Mean Squared Error."""
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    return tf.sqrt(mse).numpy()


def calculate_mape(y_true: np.array, y_pred: np.array) -> float:
    """Calculate Mean Absolute Percentage Error."""
    return tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred).numpy()


def calculate_smape(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_pred) + np.abs(y_true)) / 2
    mask = denominator != 0
    smape = np.mean(200 * np.divide(numerator[mask], denominator[mask]))
    if np.isnan(smape):
        return 0.0
    return smape


def calculate_mase(y_true: np.array, y_pred: np.array) -> float:
    """Calculate Mean Absolute Scaled Error."""
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))
    if mae_naive_no_season == 0:
        return 0.0
    else:
        return mae.numpy() / mae_naive_no_season.numpy()


def calculate_r2(y_true: np.array, y_pred: np.array) -> float:
    """Calculate R-squared (coefficient of determination)."""
    total_error = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    unexplained_error = tf.reduce_sum(tf.square(y_true - y_pred))
    r2 = 1 - unexplained_error / (total_error + tf.keras.backend.epsilon())
    return r2.numpy()


def evaluate_preds(y_true: np.array, y_pred: np.array) -> dict:
    """
    Evaluate regression predictions using various metrics.

    Parameters
    ----------
    y_true : np.array
        The true values.

    y_pred : np.array
        The predicted values.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics:
        - "mae": Mean Absolute Error,
        - "mse": Mean Squared Error,
        - "rmse": Root Mean Squared Error,
        - "mape": Mean Absolute Percentage Error,
        - "smape": Symmetric Mean Absolute Percentage Error,
        - "mase": Mean Absolute Scaled Error,
        - "r2_score": R-squared.
    """
    # Make sure float32 (for metric calculations)
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Calculate various metrics
    mae = calculate_mae(y_true, y_pred)
    mse = calculate_mse(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    smape = calculate_smape(y_true, y_pred)
    mase = calculate_mase(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)

    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "smape": smape, "mase": mase, "r2_score": r2}
