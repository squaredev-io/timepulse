import tensorflow as tf
from datetime import datetime
from src.utils import create_model_checkpoint

class MultivariateDenseModel:
    def __init__(self, input_shape, horizon, n_neurons0=128, n_neurons1=64, dropout_rate=0.2):
        self.input_shape = input_shape
        self.horizon = horizon
        self.model_name = f"dense_model_{int(datetime.now().timestamp())}"
        self.model = None
        self.n_neurons0 = n_neurons0
        self.n_neurons1 = n_neurons1
        self.dropout_rate = dropout_rate


    def build(self):
        layers = [
            tf.keras.layers.Dense(self.n_neurons0, activation="relu", input_shape=self.input_shape)
        ]

        if self.n_neurons1 is not None:
            layers.append(tf.keras.layers.Dropout(self.dropout_rate))
            layers.append(tf.keras.layers.Dense(self.n_neurons1, activation="relu"))

        layers.append(tf.keras.layers.Dense(self.horizon))

        self.model = tf.keras.Sequential(layers, name=self.model_name)


    def compile(self, loss="mae", learning_rate=0.001):
        self.model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate))


    def fit(self, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, epochs=100, batch_size=128, verbose=0):
        self.model.fit(X_train_scaled, y_train_scaled,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 verbose=verbose,
                                 validation_data=(X_val_scaled, y_val_scaled),
                                 callbacks=[create_model_checkpoint(model_name=self.model_name)])


    def compile_and_fit(self, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, loss="mae", learning_rate=0.001):
        self.build()
        self.compile(loss, learning_rate)
        self.fit(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, epochs=100, batch_size=128, verbose=0)


