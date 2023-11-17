import pandas as pd
import pytest
from src.models.xgboost import XGBRegressorModel
from src.processing.min_max_scaler import MinMaxScalerWrapper
from src.tests.v1.conftest import get_order_number
from src.tests.v1.test_utils import load_and_preprocess_data_pipeline, run_model


@pytest.mark.order(get_order_number("test_xgboost"))
def test_xgboost():
    X_train, y_train, X_test, y_test = load_and_preprocess_data_pipeline(
        data_path='/Users/nikosavgeros/Desktop/Projects/Sunrise Archive/data/ACO',
        location_name='Spain',
        country_code='ES', 
        place_filter='etap', 
        window_size=3,
        target_column='value', 
        splitter_column='stringency_category'
    )
    
    assert X_train.shape[0] == y_train.shape[0], "Number of training samples in X_train and y_train do not match"
    assert X_test.shape[0] == y_test.shape[0], "Number of testing samples in X_test and y_test do not match"

    model_instance = XGBRegressorModel(scaler_class=MinMaxScalerWrapper(), n_estimators=1000, max_depth=5, learning_rate=0.1, early_stopping_rounds=50)
    y_pred, result_metrics = run_model(model_instance, X_train, y_train, X_test, y_test, threshold=0.75, verbose=0)

    assert y_pred.shape == y_test.shape, "Shape mismatch between y_pred and y_test"

    expected_metrics = ['r2_score', 'mae', 'mse', 'rmse', 'mape', 'mase']
    assert all(metric in result_metrics for metric in expected_metrics), "Not all expected metrics are present in the results"
    assert all(result_metrics[metric] is not None for metric in expected_metrics), "Some metric values are None"
