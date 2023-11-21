import pandas as pd
import numpy as np
import tensorflow as tf
from timepulse.data.data_collection import fetch_holidays, fetch_stringency_index, fetch_weather
from timepulse.utils.splits import create_multivar_dataframe, create_windowed_dataframe, stratified_split_data
from timepulse.metrics.regression_metrics import evaluate_preds
from tests.v1.mock_data import create_mock_data
import os


def multi_data_pipeline(location_name, country_code, place_filter, window_size, target_column, splitter_column):
    df = create_mock_data(start_year=2016, end_year=2022)
    df = df[df['place']==place_filter]
    df['Date'] = pd.to_datetime(df[['year', 'month']].assign(day=1)) + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
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
