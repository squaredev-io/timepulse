import pytest
from tests.v1.conftest import get_order_number
from timepulse.models.lstm import LSTMWrapper
from tests.utils.pipelines import multi_data_pipeline
from timepulse.utils.models import run_model


@pytest.mark.order(get_order_number("test_lstm"))
def test_lstm():
    X_train, y_train, X_test, y_test = multi_data_pipeline(
        country_code="ES",
        place_filter="a",
        window_size=3,
        target_column="value",
        splitter_column="stringency_category",
    )
    lstm = LSTMWrapper(horizon=1, n_neurons=64, dropout_rate=0.2, input_shape=(1, X_train.shape[1]))

    y_pred, result_metrics = run_model(lstm, X_train, y_train, X_test, y_test, verbose=1)
    assert y_pred.shape == y_test.shape, "Shape mismatch between y_pred and y_test"

    expected_metrics = ["r2_score", "mae", "mse", "rmse", "mape", "smape", "mase"]
    assert all(
        metric in result_metrics for metric in expected_metrics
    ), "Not all expected metrics are present in the results"
    assert all(result_metrics[metric] is not None for metric in expected_metrics), "Some metric values are None"
