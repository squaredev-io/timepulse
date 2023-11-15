import pandas as pd


def fetch_weather(filename):
    """
    DEPRECATED: Use weather_data_retriever package instead.

    Fetches and preprocesses monthly weather data from a CSV file.

    Parameters
    ----------
    filename : str
        Path to the CSV file containing monthly weather data.

    Returns
    -------
    pd.DataFrame
        Monthly weather data with the 'avg_temperatures' column converted to Celsius.
    """
    monthly_weather_history_df = pd.read_csv(filename)
    monthly_weather_history_df['avg_temperatures'] = ((monthly_weather_history_df['avg_temperatures'] - 32) * 5/9).round(0)
    monthly_weather_history_df['avg_temperatures'] = monthly_weather_history_df['avg_temperatures'].astype(int)
    monthly_weather_history_df['Date'] = pd.to_datetime(monthly_weather_history_df['Date'])
    monthly_weather_history_df = monthly_weather_history_df.set_index('Date')
    return monthly_weather_history_df