import tensorflow as tf
from src.metrics import mean_absolute_scaled_error, r2_score
import numpy as np

def evaluate_preds_for_large_horizon(y_true, y_pred):
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


def make_train_test_splits(df):
    # presa_values = differenced_series['value'].values
    etap_values = df['value'].values
    timesteps = df.index.values

    # Create train and test splits the right way for time series data
    split_size = int(0.9 * len(etap_values)) # 80% train, 20% test

    # Create train data splits (everything before the split)
    X_train, y_train = timesteps[:split_size], etap_values[:split_size]

    # Create test data splits (everything after the split)
    X_test, y_test = timesteps[split_size:], etap_values[split_size:]

    return X_train, X_test, y_train, y_test


def get_labelled_windows(x, horizon=1):
    """
    # Create function to label windowed data
    Creates labels for windowed dataset.

    E.g. if horizon=1 (default)
    Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
    """
    return x[:, :-horizon], x[:, -horizon:]


def make_windows(x, window_size=7, horizon=1):
    """
    # Create function to view NumPy arrays as windows 
    Turns a 1D array into a 2D array of sequential windows of window_size.
    """
    # 1. Create a window of specific window_size (add the horizon on the end for later labelling)
    window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
    # print(f"Window step:\n {window_step}")

    # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
    window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T # create 2D array of windows of size window_size
    # print(f"Window indexes:\n {window_indexes[:3], window_indexes[-3:], window_indexes.shape}")

    # 3. Index on the target array (time series) with 2D array of multiple window steps
    windowed_array = x[window_indexes]

    # 4. Get the labelled windows
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

    return windows, labels


def make_train_test_splits(windows, labels, test_split=0.1):
    """
    # Make the train/test splits
    Splits matching pairs of windows and labels into train and test splits.
    """
    split_size = int(len(windows) * (1-test_split)) # this will default to 90% train/10% test
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    return train_windows, test_windows, train_labels, test_labels


def make_window_splits(values, size=10, horizon=1):
    full_windows, full_labels = make_windows(values, window_size=size, horizon=horizon)
    len(full_windows), len(full_labels)

    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)
    len(train_windows), len(test_windows), len(train_labels), len(test_labels)

