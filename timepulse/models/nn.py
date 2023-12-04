import tensorflow as tf
from timepulse.utils.models import create_early_stopping
from timepulse.processing.min_max_scaler import MinMaxScalerWrapper
from typing import List, Type
import numpy as np


class MultivariateDenseWrapper:
    def __init__(
        self,
        horizon: int = 1,
        n_neurons0: int = 128,
        n_neurons1: int = 64,
        dropout_rate: float = 0.2,
        epochs: int = 100,
        batch_size: int = 128,
        scaler_class: Type = MinMaxScalerWrapper(),
        callbacks: List = [create_early_stopping()],
    ) -> None:
        self.horizon = horizon
        self.n_neurons0 = n_neurons0
        self.n_neurons1 = n_neurons1
        self.dropout_rate = dropout_rate
        self.scaler_X = scaler_class
        self.scaler_y = scaler_class
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_name = f"dense_model"
        self.model = None
        self.callbacks = callbacks

    def build(self) -> None:
        layers = [tf.keras.layers.Dense(self.n_neurons0, activation="relu")]
        if self.n_neurons1 is not None:
            layers.append(tf.keras.layers.Dropout(self.dropout_rate))
            layers.append(tf.keras.layers.Dense(self.n_neurons1, activation="relu"))
        layers.append(tf.keras.layers.Dense(self.horizon))
        self.model = tf.keras.Sequential(layers, name=self.model_name)

    def compile(self, loss: str = "mae", learning_rate: float = 0.001) -> None:
        self.model.compile(loss=loss, optimizer=tf.keras.optimizers.legacy.Adam(learning_rate))

    def fit(self, X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array, verbose: int = 0) -> None:
        if self.scaler_X is not None:
            X_train = self.scaler_X.fit_transform_X(X_train)
            y_train = self.scaler_y.fit_transform_y(y_train.reshape(len(y_train), 1)).flatten()
            X_val = self.scaler_X.transform_X(X_val)
            y_val = self.scaler_y.transform_y(y_val.reshape(len(y_val), 1)).flatten()
        self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=verbose,
            validation_data=(X_val, y_val),
            callbacks=self.callbacks,
        )

    def predict(self, X_test: np.array) -> np.array:
        if self.scaler_X is not None:
            X_test = self.scaler_X.transform_X(X_test)
            y_pred = self.model.predict(X_test)
            y_pred = self.scaler_y.inverse_transform_y(y_pred.reshape(-1, 1))
        else:
            y_pred = self.model.predict(X_test)
        return y_pred.flatten()
