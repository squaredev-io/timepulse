from src.utils.timestamps import get_timestamp, get_old_timestamp
import pandas as pd
from datetime import datetime, timedelta
import random


forecasts_mock_data = [
    {"date": get_timestamp(), "place": "gibraltar", "value": 37, "confidence": 0.9},
    {"date": get_timestamp(), "place": "rguadalmansa", "value": 3417, "confidence": 0.6},
    {"date": get_timestamp(), "place": "rguadalmansa", "value": 9867, "confidence": 0.7},
    {"date": get_timestamp(), "place": "si4", "value": 723, "confidence": 0.8},
]


format = "%Y-%m-%d %H:%M:%S"
hourly_mock_data = pd.DataFrame(
    {
        "place": ["A"] * 100,
        "date": [
            (datetime(2023, 1, 1) + timedelta(hours=i + i % 7)).strftime(format) for i in range(100)
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
            (datetime(2023, 1, 1) + timedelta(days=i + i % 5)).strftime(format) for i in range(100)
        ],
        "value": [random.random() for _ in range(100)],
    }
)

weekly_mock_data = pd.DataFrame(
    {
        "place": ["A"] * 100,
        "date": [
            (datetime(2023, 1, 1) + timedelta(weeks=i + i % 3)).strftime(format) for i in range(100)
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
        + [(datetime(2023, 1, 1) + timedelta(days=i + i % 5)).strftime(format) for i in range(20)]
        + [(datetime(2023, 1, 1) + timedelta(weeks=i + i % 3)).strftime(format) for i in range(20)]
        + [
            (datetime(2023, 1, 1) + timedelta(days=30 * i + i % 10)).strftime(format)
            for i in range(40)
        ],
        "value": [random.random() for _ in range(100)],
    }
)


def create_with_nonexistent_user_response(_id):
    return {
        "error": f'Key (user_id)=({_id}) is not present in table "users".',
        "status_code": 400,
    }


def create_with_nonexistent_item_response(_id):
    return {
        "error": f'Key (item_id)=({_id}) is not present in table "items".',
        "status_code": 400,
    }


def user_not_found_response(_id):
    return {f"error": f"User with id: {_id} not found.", "status_code": 404}


def item_not_found_response(_id):
    return {f"error": f"Item with id: {_id} not found.", "status_code": 404}
