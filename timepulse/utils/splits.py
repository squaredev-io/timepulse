import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Union, Tuple


def create_multivar_dataframe(base_df: pd.DataFrame, *additional_dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Create a multivariate DataFrame by merging a base DataFrame with additional DataFrames based on a date range.

    Parameters:
    - base_df (pd.DataFrame): The base DataFrame.
    - additional_dfs (list of pd.DataFrame): List of additional DataFrames to be merged based on the date range.

    Returns:
    - multivar_df (pd.DataFrame): The resulting multivariate DataFrame.

    Examples:
    - multivar_df = create_multivar_dataframe(df, monthly_strigency_index_df, monthly_holidays_df, monthly_weather_history_df)
    """
    # Find the minimum and maximum dates from the base DataFrame
    min_date = base_df.index.min()
    max_date = base_df.index.max()

    # Create a monthly date range with the last day of each month
    date_range = pd.date_range(start=min_date, end=max_date, freq="M")

    # Create a DataFrame with the date range to ensure complete coverage
    complete_date_range_df = pd.DataFrame({"Date": date_range})
    complete_date_range_df = complete_date_range_df.set_index(["Date"])

    # Initialize the multivariate DataFrame with the base DataFrame
    multivar_df = base_df

    # Merge additional DataFrames based on the date range
    for additional_df in additional_dfs:
        # Merge with the complete date range DataFrame to ensure all dates are included
        merged_df = pd.merge(complete_date_range_df, additional_df, on="Date", how="left")
        # Fill missing values with 0 and convert to integer
        merged_df = merged_df.fillna(0).astype(int)
        # Perform the merge with the multivariate DataFrame
        multivar_df = pd.merge(multivar_df, merged_df, left_index=True, right_index=True, how="left")

    return multivar_df


def create_windowed_dataframe(base_df: pd.DataFrame, target_column: str, window_size: int = 3) -> pd.DataFrame:
    """
    Create a windowed DataFrame by shifting values of a specified column.

    Parameters:
    - base_df (pd.DataFrame): The base DataFrame.
    - target_column (str): The name of the column to create windowed features for.
    - window_size (int): The size of the window.

    Returns:
    - windowed_df (pd.DataFrame): The resulting windowed DataFrame.

    Examples:
    - multivar_df = create_windowed_dataframe(base_df=df, target_column='value', window_size=3)
    """
    # Copy the subset of the DataFrame to avoid SettingWithCopyWarning
    windowed_df = base_df.copy()

    # Add windowed columns
    for i in range(window_size):
        # Use loc to modify the copied DataFrame
        windowed_df[f"{target_column}-{i+1}"] = windowed_df[target_column].shift(periods=i + 1)

    # Drop rows with NaN values
    windowed_df = windowed_df.dropna()

    return windowed_df


def get_labelled_windows(x: np.array, horizon: int = 1) -> Tuple[np.array, np.array]:
    """
    Creates labels for windowed dataset.

    E.g. if horizon=1 (default)
    Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
    """
    return x[:, :-horizon], x[:, -horizon:]


def make_windows(x: np.array, window_size: int = 7, horizon: int = 1) -> Tuple[np.array, np.array]:
    """
    Create function to view NumPy arrays as windows.
    Turns a 1D array into a 2D array of sequential windows of window_size.
    """
    # 1. Create a window of specific window_size (add the horizon on the end for later labelling)
    window_step = np.expand_dims(np.arange(window_size + horizon), axis=0)

    # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
    window_indexes = window_step + np.expand_dims(np.arange(len(x) - (window_size + horizon - 1)), axis=0).T

    # 3. Index on the target array (time series) with 2D array of multiple window steps
    windowed_array = x[window_indexes]

    # 4. Get the labelled windows
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

    return windows, labels


def make_train_test_splits(windows: np.array, labels: np.array, test_split: float = 0.1):
    """
    Splits matching pairs of windows and labels into train and test splits.
    """
    split_size = int(len(windows) * (1 - test_split))
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    return train_windows, test_windows, train_labels, test_labels


def make_window_splits(values: np.array, size: int = 10, horizon: int = 1):
    """
    Splits a time series into input windows and corresponding labels.
    """
    full_windows, full_labels = make_windows(values, window_size=size, horizon=horizon)

    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

    return train_windows, test_windows, train_labels, test_labels
