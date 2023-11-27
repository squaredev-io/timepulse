import pandas as pd
import re, os
import holidays
from typing import Literal, List
import timepulse


def fetch_stringency_index(country: Literal["Italy", "Spain"]) -> pd.DataFrame:
    """
    Fetches and preprocesses monthly stringency index data for the specified country.

    Parameters
    ----------
    country : Literal
        Country name for which the stringency index data is fetched.

    Returns
    -------
    pd.DataFrame
        Monthly stringency index data with a calculated mode for 'stringency_category'.

    Example
    -------
    monthly_strigency_index_df = fetch_stringency_index('Spain')
    """
    stringency_index_avg_url = (
        "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/timeseries/stringency_index_avg.csv"
    )
    spain_strigency_index_df = pd.read_csv(stringency_index_avg_url)
    spain_strigency_index_df = spain_strigency_index_df[spain_strigency_index_df["country_name"] == country]

    # Define a regular expression pattern to match the date format
    date_pattern = r"(\d{2})([A-Za-z]{3})(\d{4})"
    date_columns = [col for col in spain_strigency_index_df.columns if re.fullmatch(date_pattern, col)]

    spain_strigency_index_df = spain_strigency_index_df[date_columns]
    spain_strigency_index_df = spain_strigency_index_df.fillna(0)

    # Melt the DataFrame to unpivot it
    spain_strigency_index_df = pd.melt(
        spain_strigency_index_df, id_vars=None, var_name="Date", value_name="stringency_index"
    )

    # Convert the 'Date' column to a datetime format in 'yyyy-mm-dd' format
    spain_strigency_index_df["Date"] = pd.to_datetime(spain_strigency_index_df["Date"], format="%d%b%Y").dt.strftime(
        "%Y-%m-%d"
    )
    spain_strigency_index_df["Date"] = pd.to_datetime(spain_strigency_index_df["Date"])

    # Optionally, sort the DataFrame by date
    spain_strigency_index_df = spain_strigency_index_df.sort_values(by="Date")
    spain_strigency_index_df = spain_strigency_index_df.set_index("Date")

    # Define bin edges and labels
    bin_edges = [-1, 33.0, 66.0, 110.0]  # Example boundaries for low, medium, high
    bin_labels = [0, 1, 2]

    # Create the 'stringency_category' column
    spain_strigency_index_df["stringency_category"] = pd.cut(
        spain_strigency_index_df["stringency_index"], bins=bin_edges, labels=bin_labels
    )

    # Resample to monthly and calculate mean for 'stringency_index' and mode for 'stringency_category'
    monthly_strigency_index_df = spain_strigency_index_df.resample("M").agg(
        {"stringency_index": "mean", "stringency_category": lambda x: x.mode().iloc[0]}
    )

    # Drop 'stringency_index' column
    monthly_strigency_index_df.drop("stringency_index", axis=1, inplace=True)

    return monthly_strigency_index_df


def fetch_holidays(years: List, country_code: Literal["IT", "ES"]) -> pd.DataFrame:
    """
    Fetches and preprocesses monthly holiday data for the specified country.

    Parameters
    ----------
    years : List
        List of years for which holiday data is fetched.

    country_code : Literal
        Country code used to retrieve holidays.

    Returns
    -------
    pd.DataFrame
        Monthly holiday data with the total count of holidays for each month.

    Example
    -------
    years=[2020, 2021, 2022]
    monthly_holidays_df = fetch_holidays(years=years, country_code='ES')
    """
    country_holidays = holidays.CountryHoliday(country_code, observed=True, years=years)
    dates = []
    names = []
    for date, name in sorted(country_holidays.items()):
        dates.append(date)
        names.append(name)

    holidays_df = pd.DataFrame({"Date": dates, "name": names})

    holidays_df["Date"] = pd.to_datetime(holidays_df["Date"])
    holidays_df["month"] = holidays_df["Date"].dt.month
    holidays_df["year"] = holidays_df["Date"].dt.year
    monthly_holidays_df = holidays_df.groupby(["year", "month"]).size().reset_index(name="total_holidays")
    last_dates_of_month = monthly_holidays_df.apply(
        lambda row: pd.Timestamp(row["year"], row["month"], 1) + pd.offsets.MonthEnd(), axis=1
    )
    monthly_holidays_df["last_date_of_month"] = last_dates_of_month
    monthly_holidays_df = monthly_holidays_df[["last_date_of_month", "total_holidays"]]
    monthly_holidays_df.columns = ["Date", "total_holidays"]
    monthly_holidays_df = monthly_holidays_df.set_index("Date")
    return monthly_holidays_df


def load_weather(country: Literal["italy", "spain"] = "spain") -> pd.DataFrame:
    path = timepulse.__path__[0]
    df = pd.read_csv(f"{path}/data/datasets/weather_data_{country}.csv", index_col="Date")
    df.index = pd.to_datetime(df.index)
    return df
