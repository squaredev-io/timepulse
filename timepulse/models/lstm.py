from timepulse.utils.models import create_early_stopping
import tensorflow as tf
import pandas as pd
import numpy as np
from typing import List, Optional


class LSTM:
    def __init__(self, window_size: int, horizon: int = 1, epochs: int = 100, batch_size: Optional[int] = None, callbacks: List = [create_early_stopping()]) -> None:
        self.horizon = horizon
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.model_name = "lstm_model"
        self.callbacks = callbacks

    def build(self) -> None:
        inputs = tf.keras.layers.Input(shape=(self.window_size))
        x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(
            inputs
        )
        x = tf.keras.layers.LSTM(128, activation="relu")(x)
        output = tf.keras.layers.Dense(self.horizon)(x)
        self.model = tf.keras.Model(
            inputs=inputs, outputs=output, name=self.model_name
        )

    def compile(self, loss: str = "mae", learning_rate: float = 0.001) -> None:
        self.model.compile(loss=loss, optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate))

    def fit(self, X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array, verbose: int = 0) -> None:
        self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            verbose=verbose,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=self.callbacks,
        )

    def predict(self, values: np.array) -> np.array:
        return self.model.predict(values)

    def report(self) -> None:
        self.model.summary()
