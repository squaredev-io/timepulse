import tensorflow as tf

def r2_score(y_true, y_pred):
    total_error = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    unexplained_error = tf.reduce_sum(tf.square(y_true - y_pred))
    r2 = 1 - unexplained_error / (total_error + tf.keras.backend.epsilon())
    return r2
