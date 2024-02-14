import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime, timedelta
import random


def create_mock_data(
    start_year=2016, end_year=2022, places=["a", "b", "c", "d", "e", "f", "g"]
):
    """
    Generate mock data for specified years, months, and places.

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
        "a": {
            "mean": 3794562.457875,
            "std": 1572850.7034559783,
            "count": 80,
            "min": 1614366.0,
            "max": 7230859.2,
        },
        "b": {
            "mean": 3743471.0,
            "std": 1529672.2210805719,
            "count": 80,
            "min": 1613450.0,
            "max": 6844340.0,
        },
        "c": {
            "mean": 465997.7848101266,
            "std": 374236.3678958305,
            "count": 79,
            "min": 0.0,
            "max": 1286498.0,
        },
        "d": {
            "mean": 389683.2125,
            "std": 133699.41973901461,
            "count": 80,
            "min": 137743.0,
            "max": 676129.0,
        },
        "e": {"mean": 0.0, "std": 0.0, "count": 80, "min": 0.0, "max": 0.0},
        "f": {
            "mean": 48711.2625,
            "std": 118549.19224126669,
            "count": 80,
            "min": 0.0,
            "max": 535042.0,
        },
        "g": {
            "mean": 1478.0375,
            "std": 9437.022339919808,
            "count": 80,
            "min": 0.0,
            "max": 69600.0,
        },
    }

    year_range = list(range(start_year, end_year + 1))
    month_range = list(range(1, 13))
    all_combinations = list(product(month_range, year_range, places))
    df = pd.DataFrame(all_combinations, columns=["month", "year", "place"])
    df = df.sort_values(by=["year", "month", "place"])
    values = []
    for idx, row in df.iterrows():
        place = row["place"]
        value = np.random.normal(
            loc=statistics[place]["mean"], scale=statistics[place]["std"], size=1
        )
        value = np.clip(value, statistics[place]["min"], statistics[place]["max"])
        values.append(round(value[0], 1))
    df["value"] = values
    return df


format = "%Y-%m-%d %H:%M:%S"
hourly_mock_data = pd.DataFrame(
    {
        "place": ["A"] * 100,
        "date": [
            (datetime(2023, 1, 1) + timedelta(hours=i + i % 7)).strftime(format)
            for i in range(100)
        ],
        "value": [random.random() for _ in range(100)],
    }
)

long_hourly_mock_data = pd.DataFrame(
    {
        "place": ["A"] * 10000,
        "date": [
            (datetime(2023, 1, 1) + timedelta(hours=i + i % 7)).strftime(format)
            for i in range(10000)
        ],
        "value": [random.random() for _ in range(10000)],
    }
)

daily_mock_data = pd.DataFrame(
    {
        "place": ["A"] * 100,
        "date": [
            (datetime(2023, 1, 1) + timedelta(days=i + i % 5)).strftime(format)
            for i in range(100)
        ],
        "value": [random.random() for _ in range(100)],
    }
)

weekly_mock_data = pd.DataFrame(
    {
        "place": ["A"] * 100,
        "date": [
            (datetime(2023, 1, 1) + timedelta(weeks=i + i % 3)).strftime(format)
            for i in range(100)
        ],
        "value": [random.random() for _ in range(100)],
    }
)

monthly_mock_data = pd.DataFrame(
    {
        "place": ["A"] * 100,
        "date": [
            (datetime(2023, 1, 1) + timedelta(days=30 * i + i % 10)).strftime(format)
            for i in range(100)
        ],
        "value": [random.random() for _ in range(100)],
    }
)


# Define the places and the corresponding date ranges
places = ["A", "B", "C"]
date_ranges = {
    "A": pd.date_range(start=datetime(2023, 1, 1), periods=100, freq="30D"),
    "B": pd.date_range(start=datetime(2023, 2, 1), periods=100, freq="30D"),
    "C": pd.date_range(start=datetime(2023, 3, 1), periods=100, freq="30D"),
}

# Generate data for each place
data_frames = []
for place in places:
    data = {
        "place": [place] * 100,
        "date": [date.strftime(format) for date in date_ranges[place]],
        "value": [random.random() for _ in range(100)],
    }
    df = pd.DataFrame(data)
    data_frames.append(df)

# Concatenate the data frames
multiple_place_monthly_mock_data = pd.concat(data_frames, ignore_index=True)


# Define the places and the corresponding date ranges for hourly data
places = ["A", "B", "C"]
date_ranges = {
    "A": pd.date_range(start=datetime(2023, 1, 1), periods=1000, freq="1H"),
    "B": pd.date_range(start=datetime(2023, 2, 1), periods=1000, freq="1H"),
    "C": pd.date_range(start=datetime(2023, 3, 1), periods=1000, freq="1H"),
}

# Generate data for each place
data_frames = []
for place in places:
    data = {
        "place": [place] * 1000,
        "date": [date.strftime(format) for date in date_ranges[place]],
        "value": [random.random() for _ in range(1000)],
    }
    df = pd.DataFrame(data)
    data_frames.append(df)

# Concatenate the data frames
multiple_place_hourly_mock_data = pd.concat(data_frames, ignore_index=True)


# Define the places and the corresponding date ranges for hourly data
places = ["A", "B", "C"]
date_ranges = {
    "A": pd.date_range(start=datetime(2023, 1, 1), periods=100, freq="1D"),
    "B": pd.date_range(start=datetime(2023, 2, 1), periods=100, freq="1D"),
    "C": pd.date_range(start=datetime(2023, 3, 1), periods=100, freq="1D"),
}

# Generate data for each place
data_frames = []
for place in places:
    data = {
        "place": [place] * 100,
        "date": [date.strftime(format) for date in date_ranges[place]],
        "value": [random.random() for _ in range(100)],
    }
    df = pd.DataFrame(data)
    data_frames.append(df)

# Concatenate the data frames
multiple_place_daily_mock_data = pd.concat(data_frames, ignore_index=True)


mixed_interval_mock_data = pd.DataFrame(
    {
        "place": ["A"] * 100,
        "date": [
            (datetime(2023, 1, 1) + timedelta(days=30 * i + i % 10)).strftime(format)
            for i in range(20)
        ]
        + [
            (datetime(2023, 1, 1) + timedelta(days=i + i % 5)).strftime(format)
            for i in range(20)
        ]
        + [
            (datetime(2023, 1, 1) + timedelta(weeks=i + i % 3)).strftime(format)
            for i in range(20)
        ]
        + [
            (datetime(2023, 1, 1) + timedelta(days=30 * i + i % 10)).strftime(format)
            for i in range(40)
        ],
        "value": [random.random() for _ in range(100)],
    }
)
