import tensorflow as tf

def r2_score(y_true, y_pred):
    """
    Computes the R-squared (coefficient of determination) score between true and predicted values.

    R-squared is a measure of how well the predicted values match the variability of the true values.
    It ranges from 0 to 1, with 1 indicating a perfect match.

    Parameters
    ----------
    y_true : tf.Tensor
        The true values.

    y_pred : tf.Tensor
        The predicted values.

    Returns
    -------
    tf.Tensor
        The R-squared score.
    """
    total_error = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    unexplained_error = tf.reduce_sum(tf.square(y_true - y_pred))
    r2 = 1 - unexplained_error / (total_error + tf.keras.backend.epsilon())
    return r2
