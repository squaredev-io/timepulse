import pandas as pd
import numpy as np
from src.data.data_collection import fetch_holidays,fetch_stringency_index, fetch_weather
from src.processing.min_max_scaler import MinMaxScalerWrapper
from src.utils.splits import create_multivar_dataframe, create_windowed_dataframe, stratified_split_data
from src.metrics.evaluate_preds import evaluate_preds
import tensorflow as tf
from itertools import product


def create_aco_mock_data(start_year=2016, end_year=2022, places=['presa', 'etap', 'desaladora', 'rfuengirola', 'rguadalmansa', 'gibraltar', 'si4']):
    """
    Generate mock ACO (water treatment) data for specified years, months, and places.

    Parameters:
    - start_year (int): The start year for the data range (default: 2016).
    - end_year (int): The end year for the data range (inclusive) (default: 2023).
    - places (list): List of places for which to generate data (default: ACO places).

    Returns:
    - pd.DataFrame: A DataFrame containing mock ACO data with columns 'month', 'year', 'place', and 'value'.

    Example:
    - mock_df = create_aco_mock_data(2016, 2022)
    """
    statistics = {
        'presa': {'mean': 3794562.457875, 'std': 1572850.7034559783, 'count': 80, 'min': 1614366.0, 'max': 7230859.2},
        'etap': {'mean': 3743471.0, 'std': 1529672.2210805719, 'count': 80, 'min': 1613450.0, 'max': 6844340.0},
        'desaladora': {'mean': 465997.7848101266, 'std': 374236.3678958305, 'count': 79, 'min': 0.0, 'max': 1286498.0},
        'rfuengirola': {'mean': 389683.2125, 'std': 133699.41973901461, 'count': 80, 'min': 137743.0, 'max': 676129.0},
        'rguadalmansa': {'mean': 0.0, 'std': 0.0, 'count': 80, 'min': 0.0, 'max': 0.0},
        'gibraltar': {'mean': 48711.2625, 'std': 118549.19224126669, 'count': 80, 'min': 0.0, 'max': 535042.0},
        'si4': {'mean': 1478.0375, 'std': 9437.022339919808, 'count': 80, 'min': 0.0, 'max': 69600.0}
    }

    year_range = list(range(start_year, end_year+1))
    month_range = list(range(1, 13))
    all_combinations = list(product(month_range, year_range, places))
    df = pd.DataFrame(all_combinations, columns=['month', 'year','place'])
    df = df.sort_values(by=['year', 'month', 'place'])
    values = []
    for idx, row in df.iterrows():
        place = row['place']
        value = np.random.normal(loc=statistics[place]['mean'], scale=statistics[place]['std'], size=1)
        value = np.clip(value, statistics[place]['min'], statistics[place]['max'])
        values.append(round(value[0], 1))
    df['value'] = values
    return df


def preprocess_aco_data(df):
    """
    Load and preprocess ACO data from a parquet file.

    Parameters:
    - df (pd.DataFrame): Initial ACO DataFrame.

    Returns:
    - df (pd.DataFrame): Preprocessed DataFrame with dates as the index.
    """
    # Handle missing values by backward filling
    df['value'] = df['value'].bfill()

    # Create dates as index
    df['Date'] = pd.to_datetime(df[['year', 'month']].assign(day=1)) + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


def load_and_preprocess_aco_data(parquet_file):
    """
    Load and preprocess ACO data from a parquet file.

    Parameters:
    - parquet_file (str): Path to the parquet file.

    Returns:
    - df (pd.DataFrame): Preprocessed DataFrame with dates as the index.
    """
    # Read parquet file
    df = pd.read_parquet(parquet_file)

    # Handle missing values by backward filling
    df['value'] = df['value'].bfill()

    # Create dates as index
    df['Date'] = pd.to_datetime(df[['year', 'month']].assign(day=1)) + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


def load_and_preprocess_data_pipeline(data_path, location_name, country_code, place_filter, window_size, target_column, splitter_column):
    # df = load_and_preprocess_aco_data(f'{data_path}/water_network_inflow.parquet')
    df = create_aco_mock_data(start_year=2016, end_year=2022)
    df = preprocess_aco_data(df)
    df = df[df['place']==place_filter]
    df = df[['value']]
    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date = df.index.max().strftime('%Y-%m-%d')
    years = df.index.year.unique()
    holidays_df = fetch_holidays(years=years, country_code=country_code)
    stringency_index_df = fetch_stringency_index(country_code)
    weather_history_df = fetch_weather(location_name=location_name, start_date=start_date, end_date=end_date)

    multivar_df = create_multivar_dataframe(df, stringency_index_df, holidays_df, weather_history_df)
    multivar_df = create_windowed_dataframe(base_df=multivar_df,
                                            target_column=target_column,
                                            window_size=window_size)
                                            
    X_train, y_train, X_test, y_test = stratified_split_data(data=multivar_df, target_column=target_column, splitter_column=splitter_column)

    return X_train, y_train, X_test, y_test


def run_model(model_instance, X_train, y_train, X_test, y_test, threshold=0.75, verbose=0):
    model_instance.build()
    if hasattr(model_instance, 'compile'):
        model_instance.compile()
    model_instance.fit(X_train, y_train, X_test, y_test)
    y_pred = model_instance.predict(X_test)

    result_metrics = evaluate_preds(y_pred=y_pred, y_true=y_test)

    if verbose:
        print(f"\nTest results for {model_instance.model_name}:\n")
        print(f"R-squared Score: {result_metrics['r2']:.4f}")
        print(f"Mean Absolute Error: {result_metrics['mae']:.4f}")
        print(f"Mean Squared Error: {result_metrics['mse']:.4f}")
        print(f"Root Mean Squared Error: {result_metrics['rmse']:.4f}")
        print(f"Mean Absolute Percentage Error: {result_metrics['mape']:.4f}")
        print(f"Mean Absolute Scaled Error: {result_metrics['mase']:.4f}")
        print("\n")
    return y_pred, result_metrics




