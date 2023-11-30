import pandas as pd
import re
import holidays
from typing import Literal, List


def fetch_stringency_index(country: Literal["Italy", "Spain"], period: Literal["D", "M"] = "M") -> pd.DataFrame:
    """
    Fetches and preprocesses the stringency index data for the specified country.

    Parameters
    ----------
    country : Literal["Italy", "Spain"]
        The country for which the stringency index data is fetched.

    period : Literal["D", "M"], optional
        The time period for resampling. Use "D" for daily and "M" for monthly. Default is "M".

    Returns
    -------
    pd.DataFrame
        Processed stringency index data with the stringency category for each date.

    Example
    -------
    >>> fetch_stringency_index(country="Italy", period="M")
    """
    stringency_index_avg_url = (
        "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/timeseries/stringency_index_avg.csv"
    )
    strigency_index_df = pd.read_csv(stringency_index_avg_url)
    strigency_index_df = strigency_index_df[strigency_index_df["country_name"] == country]

    # Extract date columns using a regular expression pattern
    date_pattern = r"(\d{2})([A-Za-z]{3})(\d{4})"
    date_columns = [col for col in strigency_index_df.columns if re.fullmatch(date_pattern, col)]

    strigency_index_df = strigency_index_df[date_columns]
    strigency_index_df = strigency_index_df.fillna(0)

    # Unpivot the DataFrame using melt
    strigency_index_df = pd.melt(strigency_index_df, id_vars=None, var_name="Date", value_name="stringency_index")

    # Convert the 'Date' column to a datetime format in 'yyyy-mm-dd' format
    strigency_index_df["Date"] = pd.to_datetime(strigency_index_df["Date"], format="%d%b%Y").dt.strftime("%Y-%m-%d")
    strigency_index_df["Date"] = pd.to_datetime(strigency_index_df["Date"])

    # Optionally, sort the DataFrame by date
    strigency_index_df = strigency_index_df.sort_values(by="Date")
    strigency_index_df = strigency_index_df.set_index("Date")

    # Define bin edges and labels
    bin_edges = [-1, 33.0, 66.0, 110.0]  # Example boundaries for low, medium, high
    bin_labels = [0, 1, 2]

    # Create the 'stringency_category' column based on stringency_index
    strigency_index_df["stringency_category"] = pd.cut(
        strigency_index_df["stringency_index"], bins=bin_edges, labels=bin_labels
    )

    # Resample to monthly and calculate mode for 'stringency_category'
    strigency_index_df = strigency_index_df.resample(period).agg({"stringency_category": lambda x: x.mode().iloc[0]})
    strigency_index_df = strigency_index_df.fillna(0)
    return strigency_index_df


def fetch_holidays(years: List, country_code: Literal["IT", "ES"], period: Literal["D", "M"] = "M") -> pd.DataFrame:
    """
    Fetch and preprocess monthly holiday data for the specified country.

    Parameters
    ----------
    years : List[int]
        List of years for which holiday data is fetched.

    country_code : Literal["IT", "ES"]
        Country code used to retrieve holidays.

    period : Literal["D", "M"], optional
        The time period for resampling. Use "D" for daily and "M" for monthly. Default is "M".

    Returns
    -------
    pd.DataFrame
        Monthly holiday data with the total count of holidays for each month.

    Example
    -------
    >>> years = [2020, 2021, 2022]
    >>> monthly_holidays_df = fetch_holidays(years=years, country_code='ES', period='M')
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
