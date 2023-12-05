import pandas as pd
import re, os
import holidays
from typing import Literal, List


def fetch_stringency_index(country: Literal["Italy", "Spain"], period: Literal["D", "M"] = "M") -> pd.DataFrame:
    """
    Fetches and processes stringency index data for the specified country and period.

    Parameters:
    - country (Literal["Italy", "Spain"]): The name of the country for which the stringency index data is fetched.
    - period (Literal["D", "M"], optional): The time period for data resampling, either "D" for daily or "M" for monthly.
                                             Defaults to "M".

    Returns:
    - pd.DataFrame: Processed DataFrame containing the stringency index data with date-wise categories.

    Example:
    >>> fetch_stringency_index("Italy")
    """
    stringency_index_avg_url = (
        "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/timeseries/stringency_index_avg.csv"
    )
    strigency_index_df = pd.read_csv(stringency_index_avg_url)
    strigency_index_df = strigency_index_df[strigency_index_df["country_name"] == country]

    # Define a regular expression pattern to match the date format
    date_pattern = r"(\d{2})([A-Za-z]{3})(\d{4})"
    date_columns = [col for col in strigency_index_df.columns if re.fullmatch(date_pattern, col)]

    strigency_index_df = strigency_index_df[date_columns]
    strigency_index_df = strigency_index_df.fillna(0)

    strigency_index_df = pd.melt(strigency_index_df, id_vars=None, var_name="Date", value_name="stringency_index")

    strigency_index_df["Date"] = pd.to_datetime(strigency_index_df["Date"], format="%d%b%Y").dt.strftime("%Y-%m-%d")
    strigency_index_df["Date"] = pd.to_datetime(strigency_index_df["Date"])

    strigency_index_df = strigency_index_df.sort_values(by="Date")
    strigency_index_df = strigency_index_df.set_index("Date")

    # Define bin edges and labels
    bin_edges = [-1, 33.0, 66.0, 110.0]  # Example boundaries for low, medium, high
    bin_labels = [0, 1, 2]

    strigency_index_df["stringency_category"] = pd.cut(
        strigency_index_df["stringency_index"], bins=bin_edges, labels=bin_labels
    )

    strigency_index_df = strigency_index_df.resample(period).agg({"stringency_category": lambda x: x.mode().iloc[0]})
    strigency_index_df = strigency_index_df.fillna(0)

    return strigency_index_df


def fetch_holidays(years: List, country_code: Literal["IT", "ES"], period: Literal["D", "M"] = "M") -> pd.DataFrame:
    """
    Fetches and processes holidays data for the specified country and period.

    Parameters:
    - years (List): A list of years for which holidays data is fetched.
    - country_code (Literal["IT", "ES"]): The country code for the country of interest.
    - period (Literal["D", "M"], optional): The time period for data resampling, either "D" for daily or "M" for monthly.
                                             Defaults to "M".

    Returns:
    - pd.DataFrame: Processed DataFrame containing the total number of holidays for each date.

    Example:
    >>> fetch_holidays([2021, 2022], "IT")
    """
    holidays_dict = holidays.country_holidays(country_code, years=years)
    holidays_df = pd.DataFrame(
        {"total_holidays": [1 for _ in range(len(holidays_dict.keys()))]}, index=holidays_dict.keys()
    )
    holidays_df.index.name = "Date"
    holidays_df.index = pd.to_datetime(holidays_df.index)
    holidays_df = holidays_df.resample(period).agg({"total_holidays": "sum"})
    holidays_df = holidays_df.fillna(0)
    return holidays_df
