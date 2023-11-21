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

    if verbose:
        print(f"\nTest results for {model_instance.model_name}:\n")
        print(f"R-squared Score: {result_metrics['r2_score']:.4f}")
        print(f"Mean Absolute Error: {result_metrics['mae']:.4f}")
        print(f"Mean Squared Error: {result_metrics['mse']:.4f}")
        print(f"Root Mean Squared Error: {result_metrics['rmse']:.4f}")
        print(f"Mean Absolute Percentage Error: {result_metrics['mape']:.4f}")
        print(f"Mean Absolute Scaled Error: {result_metrics['mase']:.4f}")
        print("\n")
    return y_pred, result_metrics