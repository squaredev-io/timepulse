import pandas as pd
from timepulse.data.data_collection import fetch_holidays, fetch_stringency_index
from timepulse.utils.splits import create_multivar_dataframe, create_windowed_dataframe, make_train_test_splits
from tests.v1.mock_data import create_mock_data


def multi_data_pipeline(country_code, place_filter, window_size, target_column, splitter_column):
    df = create_mock_data(start_year=2016, end_year=2022)
    df = df[df["place"] == place_filter]
    df["Date"] = pd.to_datetime(df[["year", "month"]].assign(day=1)) + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df.set_index("Date", inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df[["value"]]
    years = df.index.year.unique()
    holidays_df = fetch_holidays(years=years, country_code=country_code)
    stringency_index_df = fetch_stringency_index(country_code)
    multivar_df = create_multivar_dataframe(df, stringency_index_df, holidays_df)
    multivar_df = create_windowed_dataframe(base_df=multivar_df, target_column=target_column, window_size=window_size)
    X = multivar_df.drop(["value"], axis=1)
    y = multivar_df["value"].values
    X_train, X_test, y_train, y_test = make_train_test_splits(X, y, test_split=0.2)

    return X_train, X_test, y_train, y_test
