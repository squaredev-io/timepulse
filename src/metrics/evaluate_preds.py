import tensorflow as tf
from .mase import mean_absolute_scaled_error
from .r2_score import r2_score


def evaluate_preds(y_true, y_pred):
    """
    Evaluate regression predictions using various metrics.

    Parameters
    ----------
    y_true : tf.Tensor
        The true values.

    y_pred : tf.Tensor
        The predicted values.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics: 
        - "mae": Mean Absolute Error,
        - "mse": Mean Squared Error,
        - "rmse": Root Mean Squared Error,
        - "mape": Mean Absolute Percentage Error,
        - "mase": Mean Absolute Scaled Error,
        - "r2": R-squared.
    """
    # Make sure float32 (for metric calculations)
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Calculate various metrics
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred) # puts and emphasis on outliers (all errors get squared)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {"mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy(),
            "mase": mase.numpy(),
            "r2": r2.numpy()}


def evaluate_preds_for_large_horizon(y_true, y_pred):
    """
    Evaluate regression predictions with consideration for different-sized metrics.

    Parameters
    ----------
    y_true : tf.Tensor
        The true values.

    y_pred : tf.Tensor
        The predicted values.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics:
        - "mae": Mean Absolute Error (aggregated if not scalar),
        - "mse": Mean Squared Error (aggregated if not scalar),
        - "rmse": Root Mean Squared Error (aggregated if not scalar),
        - "mape": Mean Absolute Percentage Error (aggregated if not scalar),
        - "mase": Mean Absolute Scaled Error (aggregated if not scalar),
        - "r2": R-squared (aggregated if not scalar).
    """
    # Make sure float32 (for metric calculations)
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Calculate various metrics
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Account for different sized metrics (for longer horizons, reduce to single number)
    if mae.ndim > 0: # if mae isn't already a scalar, reduce it to one by aggregating tensors to mean
      mae = tf.reduce_mean(mae)
      mse = tf.reduce_mean(mse)
      rmse = tf.reduce_mean(rmse)
      mape = tf.reduce_mean(mape)
      mase = tf.reduce_mean(mase)
      r2 = tf.reduce_mean(r2)

    return {"mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy(),
            "mase": mase.numpy(),
            "r2": r2.numpy()}
       