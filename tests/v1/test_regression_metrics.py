import pytest
from tests.v1.conftest import get_order_number
from timepulse.metrics.regression_metrics import evaluate_preds
import numpy as np
import tensorflow as tf


@pytest.mark.order(get_order_number("test_regression_metrics"))
def test_regression_metrics():
    y_true = tf.constant([2.0, 3.0, 5.0, 7.0], dtype=tf.float32)
    y_pred = tf.constant([2.5, 3.5, 4.5, 6.5], dtype=tf.float32)

    result = evaluate_preds(y_true, y_pred)
    for key in result:
        assert result[key] is not None
        assert not np.isnan(result[key])
