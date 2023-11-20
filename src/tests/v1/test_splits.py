import pytest
from src.tests.v1.conftest import get_order_number
from src.utils.splits import get_labelled_windows, make_windows, make_train_test_splits, make_window_splits
import numpy as np


@pytest.mark.order(get_order_number("test_splits"))
def test_splits():
    windowed_array = np.array([[1,2,3,4,5,6,7,8,9,10,11,12]])
    windows, horizons = get_labelled_windows(windowed_array, horizon=1)
    assert np.array_equal(windows, np.array([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]]))
    assert np.array_equal(horizons, np.array([[12]]))

    x = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
    full_windows, full_labels = make_windows(x, window_size=5, horizon=1)
    assert np.array_equal(full_windows, np.array([[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7], [4,5,6,7,8], [5,6,7,8,9], [6,7,8,9,10], [7,8,9,10,11]]))
    assert np.array_equal(full_labels, np.array([[6], [7], [8], [9], [10], [11], [12]]))

    windows = [1, 2, 3, 4, 5]
    labels = ['a', 'b', 'c', 'd', 'e']
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(windows, labels, test_split=0.2)

    assert len(train_windows) + len(test_windows) == len(windows)
    assert len(train_labels) + len(test_labels) == len(labels)

    assert len(train_windows) == len(windows) * 0.8
    assert len(test_windows) == len(windows) * 0.2

    assert train_windows + test_windows == windows
    assert train_labels + test_labels == labels

    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    window_size = 3
    horizon = 1

    train_windows, test_windows, train_labels, test_labels = make_window_splits(values, size=window_size, horizon=horizon)

    assert len(train_windows) + len(test_windows) == len(values) - (window_size + horizon) + 1
    assert len(train_labels) + len(test_labels) == len(values) - (window_size + horizon) + 1

    assert train_windows[0].shape == (window_size,)
    assert test_windows[0].shape == (window_size,)

    assert train_labels[0].shape == (horizon,)
    assert test_labels[0].shape == (horizon,)