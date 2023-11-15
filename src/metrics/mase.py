import tensorflow as tf

def mean_absolute_scaled_error(y_true, y_pred):
    """
    Computes the Mean Absolute Scaled Error (MASE) between true and predicted values.

    MASE is a metric for evaluating the accuracy of time series forecasts. It is the mean absolute error
    of the forecast divided by the mean absolute error of a naive (non-seasonal) forecast.

    Parameters
    ----------
    y_true : tf.Tensor
        The true values.

    y_pred : tf.Tensor
        The predicted values.

    Returns
    -------
    tf.Tensor
        The Mean Absolute Scaled Error (MASE).
    """
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))

    mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))

    return mae / mae_naive_no_season
