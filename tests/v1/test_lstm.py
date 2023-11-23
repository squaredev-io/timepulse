import pytest
from tests.v1.conftest import get_order_number
from timepulse.models.lstm import LSTM
from tests.utils.pipelines import multi_data_pipeline
from timepulse.utils.models import run_model


@pytest.mark.order(get_order_number("test_lstm"))
def test_lstm():
    lstm = LSTM(horizon=1, window_size=12)
    assert str(lstm.__class__) == "<class 'timepulse.models.lstm.LSTM'>"
    assert lstm.horizon == 1
    assert lstm.window_size == 12
    assert lstm.model == None

    X_train, y_train, X_test, y_test = multi_data_pipeline(
        location_name="Spain",
        country_code="ES",
        place_filter="a",
        window_size=8,
        target_column="value",
        splitter_column="stringency_category",
    )

    y_pred, result_metrics = run_model(lstm, X_train, y_train, X_test, y_test, verbose=1)
    
    assert str(type(lstm.model)) == "<class 'keras.src.engine.functional.Functional'>"
    assert y_pred.shape[0] == y_test.shape[0], "Shape mismatch between y_pred and y_test"

    expected_metrics = ["r2_score", "mae", "mse", "rmse", "mape", "smape", "mase"]
    assert all(
        metric in result_metrics for metric in expected_metrics
    ), "Not all expected metrics are present in the results"
    assert all(result_metrics[metric] is not None for metric in expected_metrics), "Some metric values are None"
