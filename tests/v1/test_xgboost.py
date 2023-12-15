import pandas as pd
import pytest
from timepulse.models.xgboost import XGBoostRegressor
from timepulse.utils.models import run_model
from timepulse.processing.standard_scaler import StandardScalerWrapper
from tests.v1.conftest import get_order_number
from tests.utils.pipelines import multi_data_pipeline
from unittest.mock import patch


@pytest.mark.order(get_order_number("test_xgboost"))
def test_xgboost():
    X_train, X_test, y_train, y_test = multi_data_pipeline(
        country_code="ES", place_filter="b", window_size=3, target_column="value", splitter_column="stringency_category"
    )

    assert X_train.shape[0] == y_train.shape[0], "Number of training samples in X_train and y_train do not match"
    assert X_test.shape[0] == y_test.shape[0], "Number of testing samples in X_test and y_test do not match"

    model_instance = XGBoostRegressor(
        scaler_class=StandardScalerWrapper(),
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.1,
        early_stopping_rounds=50,
    )

    y_pred, result_metrics = run_model(model_instance, X_train, y_train, X_test, y_test, verbose=0)
    assert y_pred.shape == y_test.shape, "Shape mismatch between y_pred and y_test"

    expected_metrics = ["r2_score", "mae", "mse", "rmse", "mape", "smape", "mase"]
    assert all(
        metric in result_metrics for metric in expected_metrics
    ), "Not all expected metrics are present in the results"
    assert all(result_metrics[metric] is not None for metric in expected_metrics), "Some metric values are None"

    default_model_instance = XGBoostRegressor(scaler_class=None)
    assert not default_model_instance.params, "The default Constructor is not called"

    y_pred, result_metrics = run_model(default_model_instance, X_train, y_train, X_test, y_test, verbose=0)
    assert y_pred.shape == y_test.shape, "Shape mismatch between y_pred and y_test"

    expected_metrics = ["r2_score", "mae", "mse", "rmse", "mape", "smape", "mase"]
    assert all(
        metric in result_metrics for metric in expected_metrics
    ), "Not all expected metrics are present in the results"
    assert all(result_metrics[metric] is not None for metric in expected_metrics), "Some metric values are None"

    with patch("joblib.dump") as mock_dump:
        default_model_instance.save()
        expected_filename = f"storage/{default_model_instance.model_name}"
        mock_dump.assert_called_once_with(default_model_instance.model, expected_filename)
