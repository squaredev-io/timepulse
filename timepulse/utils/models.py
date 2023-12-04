import tensorflow as tf
import os
import numpy as np
from typing import Tuple, Dict, Type
from timepulse.metrics.regression_metrics import evaluate_preds


def create_model_checkpoint(model_name: str, save_path: str = "storage") -> tf.keras.callbacks.ModelCheckpoint:
    """
    Creates a ModelCheckpoint callback for saving the best model during training.

    Parameters
    ----------
    model_name : str
        Name of the model.

    save_path : str, optional, default: 'storage'
        Path to save the model checkpoints.

    Returns
    -------
    tf.keras.callbacks.ModelCheckpoint
        ModelCheckpoint callback instance.
    """
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_path, model_name), verbose=0, save_best_only=True
    )


def create_early_stopping(
    monitor: str = "val_loss", patience: int = 50, restore_best_weights: bool = True
) -> tf.keras.callbacks.EarlyStopping:
    """
    Create an EarlyStopping callback for a Keras model.

    Parameters
    ----------
    monitor : str, optional
        Quantity to be monitored (default is 'val_loss').
    patience : int, optional
        Number of epochs with no improvement after which training will be stopped (default is 50).
    restore_best_weights : bool, optional
        Whether to restore model weights from the epoch with the best value of the monitored quantity (default is True).

    Returns
    -------
    tf.keras.callbacks.EarlyStopping
        EarlyStopping callback.

    Example
    -------
    early_stopping = create_early_stopping(monitor='val_loss', patience=10, restore_best_weights=True)
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor, patience=patience, restore_best_weights=restore_best_weights
    )


def run_model(
    model_instance: Type, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array, verbose=0
) -> Tuple[np.array, Dict]:
    """
    Train and evaluate a Keras model.

    Parameters
    ----------
    model_instance : Type
        An instance of any class.
    X_train : np.array
        Input features for training.
    y_train : np.array
        Target labels for training.
    X_test : np.array
        Input features for testing.
    y_test : np.array
        Target labels for testing.
    verbose : int, optional
        Verbosity mode (default is 0).

    Returns
    -------
    Tuple[np.array, dict]
        Tuple containing the model predictions and evaluation metrics.

    Example
    -------
    y_pred, result_metrics = run_model(model_instance, X_train, y_train, X_test, y_test, verbose=0)
    """
    model_instance.build()
    if hasattr(model_instance, "compile"):
        model_instance.compile()
    model_instance.fit(X_train, y_train, X_test, y_test, verbose=verbose)
    y_pred = model_instance.predict(X_test)
    result_metrics = evaluate_preds(y_pred=y_pred, y_true=y_test)

    return y_pred, result_metrics
