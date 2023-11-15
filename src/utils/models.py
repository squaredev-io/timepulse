import tensorflow as tf
import os


def create_model_checkpoint(model_name, save_path='storage'):
    """
    Creates a ModelCheckpoint callback for saving the best model during training.

    Parameters
    ----------
    model_name : str
        Name of the model.

    save_path : str, optional, default: 'storage'
        Path to save the model checkpoints.

    Returns
    -------
    tf.keras.callbacks.ModelCheckpoint
        ModelCheckpoint callback instance.
    """
    return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name),
                                            verbose=0,
                                            save_best_only=True)

