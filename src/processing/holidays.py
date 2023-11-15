import holidays
import pandas as pd


def fetch_holidays(years, country_code):
    """
    Fetches and preprocesses monthly holiday data for the specified country.

    Parameters
    ----------
    years : list
        List of years for which holiday data is fetched.

    country_code : str
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

    holidays_df = pd.DataFrame({'Date':dates, 'name':names})

    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])
    holidays_df["month"] = holidays_df['Date'].dt.month
    holidays_df["year"] = holidays_df['Date'].dt.year
    monthly_holidays_df = holidays_df.groupby(['year', 'month']).size().reset_index(name='total_holidays')
    last_dates_of_month = monthly_holidays_df.apply(lambda row: pd.Timestamp(row['year'], row['month'], 1) + pd.offsets.MonthEnd(), axis=1)
    monthly_holidays_df['last_date_of_month'] = last_dates_of_month
    monthly_holidays_df = monthly_holidays_df[['last_date_of_month', 'total_holidays']]
    monthly_holidays_df.columns = ["Date", "total_holidays"]
    monthly_holidays_df = monthly_holidays_df.set_index('Date')
    return monthly_holidays_df