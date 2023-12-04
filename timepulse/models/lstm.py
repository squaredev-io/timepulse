from timepulse.utils.models import create_early_stopping
from timepulse.processing.min_max_scaler import MinMaxScalerWrapper
import tensorflow as tf
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Type


class LSTMWrapper:
    def __init__(
        self,
        input_shape: Tuple,
        horizon: int = 1,
        n_neurons: int = 64,
        dropout_rate: float = 0.2,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        scaler_class: Type = MinMaxScalerWrapper(),
        callbacks: List = [create_early_stopping()],
    ) -> None:
        self.horizon = horizon
        self.input_shape = input_shape
        self.n_neurons = n_neurons
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler_X = scaler_class
        self.scaler_y = scaler_class
        self.model = None
        self.model_name = "lstm_model"
        self.callbacks = callbacks

    def build(self) -> None:
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(self.n_neurons, input_shape=self.input_shape),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(self.horizon),
            ]
        )

    def compile(self, loss: str = "mean_squared_error", learning_rate: float = 0.001) -> None:
        self.model.compile(loss=loss, optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate))

    def fit(self, X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array, verbose: int = 0) -> None:
        if self.scaler_X is not None:
            X_train = self.scaler_X.fit_transform_X(X_train)
            y_train = self.scaler_y.fit_transform_y(y_train.reshape(len(y_train), 1)).flatten()
            X_val = self.scaler_X.transform_X(X_val)
            y_val = self.scaler_y.transform_y(y_val.reshape(len(y_val), 1)).flatten()
        X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_val_reshaped = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
        self.model.fit(
            X_train_reshaped,
            y_train,
            epochs=self.epochs,
            verbose=verbose,
            batch_size=self.batch_size,
            validation_data=(X_val_reshaped, y_val),
            callbacks=self.callbacks,
        )

    def predict(self, X_test: np.array) -> np.array:
        if self.scaler_X is not None:
            X_test = self.scaler_X.transform_X(X_test)
            X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
            y_pred = self.model.predict(X_test_reshaped)
            y_pred = self.scaler_y.inverse_transform_y(y_pred.reshape(-1, 1))
        else:
            X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
            y_pred = self.model.predict(X_test_reshaped)
        return y_pred.flatten()
