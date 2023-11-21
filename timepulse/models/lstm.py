from timepulse.utils.models import create_model_checkpoint
import tensorflow as tf
from datetime import datetime
import os


class LSTM():
    def __init__(self, horizon, window_size):
        self.horizon = horizon
        self.window_size = window_size
        self.model = None
        

    def build(self):
        # Let's build an LSTM model with the Functional API
        inputs = tf.keras.layers.Input(shape=(self.window_size))
        x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs) # expand input dimension to be compatible with LSTM
        x = tf.keras.layers.LSTM(128, activation="relu")(x) # using the tanh loss function results in a massive error
        output = tf.keras.layers.Dense(self.horizon)(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=output, name=f"lstm_model_{datetime.now().strftime('D%Y-%m-%dT%H.%M')}")


    def compile(self, loss="mae", learning_rate=0.001):
        # Compile model
        self.model.compile(loss=loss, optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate))


    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=None, verbose=0):
        # Seems when saving the model several warnings are appearing: https://github.com/tensorflow/tensorflow/issues/47554 
        self.model.fit(X_train,
                    y_train,
                    epochs=epochs,
                    verbose=verbose,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=[create_model_checkpoint(model_name=self.model.name)])    


    def predict(self, values):
        return self.model.predict(values)
    

    def report(self):
        self.model.summary()
