import pandas as pd
import pytest
from src.models.nn import MultivariateDenseModel
from src.processing.min_max_scaler import MinMaxScalerWrapper
from src.tests.v1.conftest import get_order_number
from src.tests.utils.pipelines import load_and_preprocess_data_pipeline, run_model


@pytest.mark.order(get_order_number("test_nn"))
def test_nn():
    X_train, y_train, X_test, y_test = load_and_preprocess_data_pipeline(
        data_path='C:/Users/user/Desktop/Squaredev/Projects/Sunrise/repos/Sunrise Archive/data/ACO',
        location_name='Spain',
        country_code='ES', 
        place_filter='etap', 
        window_size=3,
        target_column='value', 
        splitter_column='stringency_category'
    )

    assert X_train.shape[0] == y_train.shape[0], "Number of training samples in X_train and y_train do not match"
    assert X_test.shape[0] == y_test.shape[0], "Number of testing samples in X_test and y_test do not match"

    model_instance = MultivariateDenseModel(horizon=1, scaler_class=MinMaxScalerWrapper())
    y_pred, result_metrics = run_model(model_instance, X_train, y_train, X_test, y_test, threshold=0.75, verbose=1)

    assert y_pred.shape == y_test.shape, "Shape mismatch between y_pred and y_test"

    expected_metrics = ['r2_score', 'mae', 'mse', 'rmse', 'mape', 'mase']
    assert all(metric in result_metrics for metric in expected_metrics), "Not all expected metrics are present in the results"
    assert all(result_metrics[metric] is not None for metric in expected_metrics), "Some metric values are None"

    model_instance_with_no_scaler = MultivariateDenseModel(horizon=1, scaler_class=None)
    y_pred, result_metrics = run_model(model_instance_with_no_scaler, X_train, y_train, X_test, y_test, threshold=0.75, verbose=1)

    assert y_pred.shape == y_test.shape, "Shape mismatch between y_pred and y_test"

    expected_metrics = ['r2_score', 'mae', 'mse', 'rmse', 'mape', 'mase']
    assert all(metric in result_metrics for metric in expected_metrics), "Not all expected metrics are present in the results"
    assert all(result_metrics[metric] is not None for metric in expected_metrics), "Some metric values are None"