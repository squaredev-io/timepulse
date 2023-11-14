import tensorflow as tf
from datetime import datetime
from src.utils import create_model_checkpoint

class MultivariateDenseModel:
    def __init__(self, horizon, n_neurons0=128, n_neurons1=64, dropout_rate=0.2):
        self.horizon = horizon
        self.n_neurons0 = n_neurons0
        self.n_neurons1 = n_neurons1
        self.dropout_rate = dropout_rate
        self.model = None


    def build(self):
        layers = [
            tf.keras.layers.Dense(self.n_neurons0, activation="relu")
        ]

        if self.n_neurons1 is not None:
            layers.append(tf.keras.layers.Dropout(self.dropout_rate))
            layers.append(tf.keras.layers.Dense(self.n_neurons1, activation="relu"))

        layers.append(tf.keras.layers.Dense(self.horizon))

        self.model = tf.keras.Sequential(layers, name=f"dense_model_{int(datetime.now().timestamp())}")


    def compile(self, loss="mae", learning_rate=0.001):
        self.model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate))


    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=128, verbose=0):
        self.model.fit(X_train, y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 verbose=verbose,
                                 validation_data=(X_val, y_val),
                                 callbacks=[create_model_checkpoint(model_name=self.model.name)])


