from src.utils import make_windows, make_train_test_splits, create_model_checkpoint
import tensorflow as tf
from datetime import datetime

class LSTM():
    def __init__(self, horizon, window_size):
        self.horizon = horizon
        self.window_size = window_size
        tf.random.set_seed(42)

    def prepare_data(self, timesteps, values):
        full_windows, full_labels = make_windows(values, window_size=self.window_size, horizon=self.horizon)
        self.train_windows, self.test_windows, self.train_labels, self.test_labels = make_train_test_splits(windows=full_windows, labels=full_labels, test_split=0.1)
        

    def build(self):
        # Let's build an LSTM model with the Functional API
        inputs = tf.keras.layers.Input(shape=(self.window_size))
        x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs) # expand input dimension to be compatible with LSTM
        x = tf.keras.layers.LSTM(128, activation="relu")(x) # using the tanh loss function results in a massive error
        output = tf.keras.layers.Dense(self.horizon)(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=output, name=f"lstm_model_{datetime.now().isoformat()}")

        # Compile model
        self.model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam())

        return self.model.summary()

    def fit(self):
        # Seems when saving the model several warnings are appearing: https://github.com/tensorflow/tensorflow/issues/47554 
        self.model.fit(self.train_windows,
                    self.train_labels,
                    epochs=100,
                    verbose=0,
                    batch_size=None,
                    validation_data=(self.test_windows, self.test_labels),
                    callbacks=[create_model_checkpoint(model_name=self.model.name)])