import tensorflow as tf
import numpy as np
from timepulse.utils.models import create_early_stopping
from typing import Tuple, List, Dict, Type
from timepulse.processing.min_max_scaler import MinMaxScalerWrapper


class NBeatsBlock(tf.keras.layers.Layer):
    """
    Custom layer for the N-Beats model, consisting of stacked fully connected layers with ReLU activation,
    followed by a theta layer with linear activation.

    Parameters
    ----------
    input_size : int
        The size of the input for the block.

    theta_size : int
        The size of the theta layer.

    horizon : int
        The forecasting horizon.

    n_neurons : int
        The number of neurons in each hidden layer.

    n_layers : int
        The number of hidden layers in the block.

    **kwargs
        Additional keyword arguments for the parent class (input_shape, trainable, name).

    Returns
    -------
    backcast : np.array
        The backcast predictions.

    forecast : np.array
        The forecast predictions.
    """

    def __init__(
        self,
        input_size: int,
        theta_size: int,
        horizon: int,
        n_neurons: int,
        n_layers: int,
        **kwargs: Dict,
    ) -> None:
        super().__init__(**kwargs)
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
        self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

    def call(self, inputs: np.array) -> Tuple[np.array, np.array]:
        """
        Forward pass through the NBeatsBlock.

        Parameters
        ----------
        inputs : np.array
            The input data.

        Returns
        -------
        Tuple[np.array, np.array]
            Containing the backcast and forecast predictions.
        """
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        theta = self.theta_layer(x)
        backcast, forecast = theta[:, : self.input_size], theta[:, -self.horizon :]
        return backcast, forecast


class NBeatsWrapper(tf.keras.Model):
    """
    N-Beats Model for time series forecasting.

    Parameters
    ----------
    window_size : int
        The size of the input window.

    horizon : int, optional, default: 1
        The forecasting horizon.

    n_neurons : int, optional, default: 512
        The number of neurons in each hidden layer.

    n_layers : int, optional, default: 4
        The number of hidden layers in each NBeatsBlock.

    n_stacks : int, optional, default: 30
        The number of stacked NBeatsBlocks in the model.

    epochs : int, optional, default: 5000
        The number of training epochs.

    batch_size : int, optional, default: 1024
        The batch size for training.

    scaler_class : Type, optional, default: MinMaxScalerWrapper
        The class used for scaling input and output data.

    callbacks : List[tf.keras.callbacks.Callback], optional, default: [ReduceLROnPlateau, EarlyStopping]
        List of callbacks to monitor and control the training process.

    **kwargs
        Additional keyword arguments for the parent class.

    Attributes
    ----------
    model : tf.keras.Model
        The NBeats model.

    Methods
    -------
    build()
        Build the NBeats model.

    compile(loss="mae", learning_rate=0.001, metrics=["mae", "mse"])
        Compile the NBeats model with specified loss function, learning rate, and metrics.

    fit(X_train, y_train, X_val, y_val, verbose=0)
        Train the NBeats model with EarlyStopping and ReduceLROnPlateau callbacks.

    predict(X_test)
        Generate predictions using the trained NBeats model.

    Examples
    --------
    >>> nbeats_instance = NBeatsWrapper(window_size=10, horizon=3)
    >>> nbeats_instance.build()
    >>> nbeats_instance.compile()
    >>> nbeats_instance.fit(X_train, y_train, X_val, y_val)
    >>> predictions = nbeats_instance.predict(test_data)
    >>> mse = nbeats_model.model.evaluate(test_data)
    """

    def __init__(
        self,
        window_size: int,
        horizon: int = 1,
        n_neurons: int = 512,
        n_layers: int = 4,
        n_stacks: int = 30,
        epochs: int = 5000,
        batch_size: int = 1024,
        scaler_class: Type = MinMaxScalerWrapper(),
        callbacks: List = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=0),
            create_early_stopping(monitor="val_loss", patience=200, restore_best_weights=True),
        ],
        **kwargs: Dict,
    ) -> None:
        super().__init__(**kwargs)
        self.input_size = window_size * horizon
        self.theta_size = window_size * horizon + horizon
        self.window_size = window_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.n_stacks = n_stacks
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_name = f"nbeats_model"
        self.model = None
        self.scaler_X = scaler_class
        self.scaler_y = scaler_class
        self.callbacks = callbacks
        self.initial_block = NBeatsBlock(
            input_size=window_size * horizon,
            theta_size=window_size * horizon + horizon,
            horizon=horizon,
            n_neurons=n_neurons,
            n_layers=n_layers,
            name="InitialBlock",
        )

    def build(self) -> None:
        stack_input = tf.keras.layers.Input(shape=(self.input_size), name="stack_input")
        backcast, forecast = self.initial_block(stack_input)
        residuals = tf.keras.layers.subtract([stack_input, backcast], name="subtract_00")
        for i in range(self.n_stacks - 1):
            backcast, block_forecast = NBeatsBlock(
                input_size=self.input_size,
                theta_size=self.theta_size,
                horizon=self.horizon,
                n_neurons=self.n_neurons,
                n_layers=self.n_layers,
                name=f"NBeatsBlock_{i}",
            )(residuals)
            residuals = tf.keras.layers.subtract([residuals, backcast], name=f"subtract_{i}")
            forecast = tf.keras.layers.add([forecast, block_forecast], name=f"add_{i}")
        self.model = tf.keras.Model(inputs=stack_input, outputs=forecast, name=self.model_name)

    def compile(self, loss: str = "mae", learning_rate: float = 0.001, metrics: List = ["mae", "mse"]) -> None:
        self.model.compile(loss=loss, optimizer=tf.keras.optimizers.legacy.Adam(learning_rate), metrics=metrics)

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
            validation_data=(X_val, y_val),
            verbose=verbose,
            batch_size=self.batch_size,
            callbacks=self.callbacks,
        )

    def predict(self, X_test: np.array) -> None:
        if self.scaler_X is not None:
            X_test = self.scaler_X.transform_X(X_test)
            y_pred = self.model.predict(X_test)
            y_pred = self.scaler_y.inverse_transform_y(y_pred.reshape(-1, 1))
        else:
            y_pred = self.model.predict(X_test)
        return y_pred.flatten()
