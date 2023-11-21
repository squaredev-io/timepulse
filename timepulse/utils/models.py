import tensorflow as tf
import os
from timepulse.metrics.regression_metrics import evaluate_preds

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


def run_model(model_instance, X_train, y_train, X_test, y_test, threshold=0.75, verbose=0):
    model_instance.build()
    if hasattr(model_instance, 'compile'):
        model_instance.compile()
    model_instance.fit(X_train, y_train, X_test, y_test, verbose=verbose)
    y_pred = model_instance.predict(X_test)
    result_metrics = evaluate_preds(y_pred=y_pred, y_true=y_test)
    
    return y_pred, result_metrics