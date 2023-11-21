import tensorflow as tf
from datetime import datetime
from timepulse.utils.models import create_model_checkpoint
from timepulse.processing.min_max_scaler import MinMaxScalerWrapper

class MultivariateDenseModel:
    def __init__(self, horizon, n_neurons0=128, n_neurons1=64, dropout_rate=0.2, scaler_class=MinMaxScalerWrapper()):
        self.horizon = horizon
        self.n_neurons0 = n_neurons0
        self.n_neurons1 = n_neurons1
        self.dropout_rate = dropout_rate
        self.scaler_X = scaler_class
        self.scaler_y = scaler_class
        self.model_name = f"dense_model"
        self.model = None

    def build(self):
        layers = [tf.keras.layers.Dense(self.n_neurons0, activation="relu")]
        if self.n_neurons1 is not None:
            layers.append(tf.keras.layers.Dropout(self.dropout_rate))
            layers.append(tf.keras.layers.Dense(self.n_neurons1, activation="relu"))
        layers.append(tf.keras.layers.Dense(self.horizon))
        self.model = tf.keras.Sequential(layers, name=self.model_name)

    def compile(self, loss="mae", learning_rate=0.001):
        self.model.compile(loss=loss, optimizer=tf.keras.optimizers.legacy.Adam(learning_rate))

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=128, verbose=0):
        if self.scaler_X is not None:
            X_train = self.scaler_X.fit_transform_X(X_train)
            y_train = self.scaler_y.fit_transform_y(y_train.reshape(len(y_train), 1)).flatten()
            X_val = self.scaler_X.transform_X(X_val)
            y_val = self.scaler_y.transform_y(y_val.reshape(len(y_val), 1)).flatten()
        self.model.fit(X_train, 
                       y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=verbose,
                        validation_data=(X_val, y_val),
                        callbacks=[create_model_checkpoint(model_name=self.model_name)])

    def predict(self, X_test):
        if self.scaler_X is not None:
            X_test = self.scaler_X.fit_transform_X(X_test)
            y_pred = self.model.predict(X_test)
            y_pred = self.scaler_y.inverse_transform_y(y_pred.reshape(-1,1))
        else:
            y_pred = self.model.predict(X_test)
        return y_pred.flatten()
    



