import pandas as pd
import os


def fetch_weather(path, csv_file):
    # Define the file path and CSV filename
    weather_history_csv_file = path + csv_file

    if not os.path.exists(weather_history_csv_file):
        from fetch_weather_data import main
        main(parquet_file=weather_history_csv_file, csv_filename=weather_history_csv_file)

    monthly_weather_history_df = pd.read_csv(weather_history_csv_file)
    monthly_weather_history_df['avg_temperatures'] = ((monthly_weather_history_df['avg_temperatures'] - 32) * 5/9).round(0)
    monthly_weather_history_df['avg_temperatures'] = monthly_weather_history_df['avg_temperatures'].astype(int)
    monthly_weather_history_df['Date'] = pd.to_datetime(monthly_weather_history_df['Date'])
    monthly_weather_history_df = monthly_weather_history_df.set_index('Date')
    monthly_weather_history_df.head()