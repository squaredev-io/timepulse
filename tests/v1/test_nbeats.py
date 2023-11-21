import pandas as pd
import pytest
from timepulse.models.nbeats import NBeats
from timepulse.utils.models import run_model
from tests.v1.conftest import get_order_number
from tests.utils.pipelines import multi_data_pipeline


@pytest.mark.order(get_order_number("test_nbeats"))
def test_nbeats():
    X_train, y_train, X_test, y_test = multi_data_pipeline(
        location_name='Spain',
        country_code='ES', 
        place_filter='a', 
        window_size=3,
        target_column='value', 
        splitter_column='stringency_category'
    )

    assert X_train.shape[0] == y_train.shape[0], "Number of training samples in X_train and y_train do not match"
    assert X_test.shape[0] == y_test.shape[0], "Number of testing samples in X_test and y_test do not match"

    model_instance = NBeats(window_size=(len(X_train.columns)), horizon=1)
    y_pred, result_metrics = run_model(model_instance, X_train, y_train, X_test, y_test, threshold=0.75, verbose=0)

    assert y_pred.shape == y_test.shape, "Shape mismatch between y_pred and y_test"

    expected_metrics = ['r2_score', 'mae', 'mse', 'rmse', 'mape', 'mase']
    assert all(metric in result_metrics for metric in expected_metrics), "Not all expected metrics are present in the results"
    assert all(result_metrics[metric] is not None for metric in expected_metrics), "Some metric values are None"