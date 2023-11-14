import tensorflow as tf
from datetime import datetime

# Create NBeatsBlock custom layer 
class NBeatsBlock(tf.keras.layers.Layer):
  def __init__(self, # the constructor takes all the hyperparameters for the layer
               input_size: int,
               theta_size: int,
               horizon: int,
               n_neurons: int,
               n_layers: int,
               **kwargs): # the **kwargs argument takes care of all of the arguments for the parent class (input_shape, trainable, name)
    super().__init__(**kwargs)
    self.input_size = input_size
    self.theta_size = theta_size
    self.horizon = horizon
    self.n_neurons = n_neurons
    self.n_layers = n_layers

    # Block contains stack of 4 fully connected layers each has ReLU activation
    self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
    # Output of block is a theta layer with linear activation
    self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

  def call(self, inputs): # the call method is what runs when the layer is called 
    x = inputs 
    for layer in self.hidden: # pass inputs through each hidden layer 
      x = layer(x)
    theta = self.theta_layer(x) 
    # Output the backcast and forecast from theta
    backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
    return backcast, forecast
  

# Custom NBeats model
class NBeats(tf.keras.Model):
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

    Attributes
    ----------
    model : tf.keras.Model
        The NBeats model.

    Methods
    -------
    prepare_data(X_train, y_train, X_val, y_val)
        Prepare data for training and validation.

    build()
        Build the NBeats model.

    compile(loss="mae", learning_rate=0.001, metrics=["mae", "mse"])
        Compile the NBeats model with specified loss function, learning rate, and metrics.

    fit(X_train, y_train, X_val, y_val, n_epochs=5000, batch_size=1024, verbose=0)
        Train the NBeats model with EarlyStopping and ReduceLROnPlateau callbacks.

    Examples
    --------
    >>> nbeats_instance = NBeats(window_size=10, horizon=3)
    >>> nbeats_instance.build()
    >>> nbeats_instance.compile()
    >>> nbeats_instance.fit(X_train, y_train, X_val, y_val)
    >>> predictions = nbeats_instance.model.predict(test_data)
    >>> mse = nbeats_model.model.evaluate(test_data)
    """
        
    def __init__(self, window_size, horizon=1, n_neurons=512, n_layers=4, n_stacks=30, **kwargs):
        super().__init__(**kwargs)
        self.input_size = window_size * horizon
        self.theta_size = window_size * horizon + horizon
        self.window_size = window_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.n_stacks = n_stacks
        self.model = None

        # Create the initial NBeatsBlock layer
        self.initial_block = NBeatsBlock(
            input_size=window_size * horizon,
            theta_size=window_size * horizon + horizon,
            horizon=horizon,
            n_neurons=n_neurons,
            n_layers=n_layers,
            name="InitialBlock"
        )


    def build(self):
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
                name=f"NBeatsBlock_{i}"
            )(residuals)
            residuals = tf.keras.layers.subtract([residuals, backcast], name=f"subtract_{i}")
            forecast = tf.keras.layers.add([forecast, block_forecast], name=f"add_{i}")
            
        self.model = tf.keras.Model(inputs=stack_input, outputs=forecast, name=f"nbeats_model_{int(datetime.now().timestamp())}")


    def compile(self, loss="mae", learning_rate=0.001, metrics=["mae", "mse"]):
        self.model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            metrics=metrics
        )


    def fit(self, X_train, y_train, X_val, y_val, n_epochs=5000, batch_size=1024, verbose=0):
        # Turn train and validation arrays into tensor Datasets
        train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train)
        train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)

        validation_features_dataset = tf.data.Dataset.from_tensor_slices(X_val)
        validation_labels_dataset = tf.data.Dataset.from_tensor_slices(y_val)

        # Combine features & labels
        train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
        validation_dataset = tf.data.Dataset.zip((validation_features_dataset, validation_labels_dataset))

        # Batch and prefetch for optimal performance
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        self.model.fit(
            train_dataset,
            epochs=n_epochs,
            validation_data=validation_dataset,
            verbose=verbose,
            batch_size=batch_size, # taken from Appendix D in N-BEATS paper
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1)
            ]
        )


