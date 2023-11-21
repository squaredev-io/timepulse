import pytest
from tests.v1.conftest import get_order_number
from timepulse.processing.min_max_scaler import MinMaxScalerWrapper
from timepulse.processing.standard_scaler import StandardScalerWrapper
import numpy as np


@pytest.mark.order(get_order_number("test_scalers"))
def test_scalers():
    
    # Min Max Scaler Wrapper
    data_X = np.array([[1, 2], [3, 4], [5, 6]])
    data_y = np.array([1, 2, 3])
    scaler_wrapper = MinMaxScalerWrapper()
    transformed_X = scaler_wrapper.fit_transform_X(data_X)
    assert np.allclose(transformed_X, np.array([[0., 0.], [0.5, 0.5], [1., 1.]]))

    transformed_X = scaler_wrapper.transform_X(data_X)
    assert np.allclose(transformed_X, np.array([[0., 0.], [0.5, 0.5], [1., 1.]]))

    inverse_transformed_X = scaler_wrapper.inverse_transform_X(transformed_X)
    assert np.allclose(inverse_transformed_X, data_X)

    transformed_y = scaler_wrapper.fit_transform_y(data_y.reshape(-1, 1))
    assert np.allclose(transformed_y, np.array([[0.], [0.5], [1.]]))

    transformed_y = scaler_wrapper.transform_y(data_y.reshape(-1, 1))
    assert np.allclose(transformed_y, np.array([[0.], [0.5], [1.]]))

    inverse_transformed_y = scaler_wrapper.inverse_transform_y(transformed_y)
    assert np.allclose(inverse_transformed_y, data_y.reshape(-1, 1))


    # Standard Scaler Wrapper
    data_X = np.array([[1, 2], [3, 4], [5, 6]])
    data_y = np.array([1, 2, 3])
    mean_X = np.mean(data_X, axis=0)
    mean_y = np.mean(data_y)
    std_X = np.std(data_X, axis=0)
    std_y = np.std(data_y)
    expected_transformed_X = (data_X - mean_X) / std_X
    expected_transformed_y = (data_y - mean_y) / std_y
    scaler_wrapper = StandardScalerWrapper()

    transformed_X = scaler_wrapper.fit_transform_X(data_X)
    assert np.allclose(transformed_X, expected_transformed_X, rtol=1e-5, atol=1e-8)

    transformed_X = scaler_wrapper.transform_X(data_X)
    assert np.allclose(transformed_X, expected_transformed_X, rtol=1e-5, atol=1e-8)

    inverse_transformed_X = scaler_wrapper.inverse_transform_X(transformed_X)
    assert np.allclose(inverse_transformed_X, data_X)

    transformed_y = scaler_wrapper.fit_transform_y(data_y.reshape(-1, 1))
    assert np.allclose(transformed_y.flatten(), expected_transformed_y, rtol=1e-5, atol=1e-8)

    transformed_y = scaler_wrapper.transform_y(data_y.reshape(-1, 1))
    assert np.allclose(transformed_y.flatten(), expected_transformed_y, rtol=1e-5, atol=1e-8)

    inverse_transformed_y = scaler_wrapper.inverse_transform_y(transformed_y)
    assert np.allclose(inverse_transformed_y, data_y.reshape(-1, 1))