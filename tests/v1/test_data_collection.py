import pytest
from tests.v1.conftest import get_order_number
from timepulse.data.data_collection import fetch_weather, fetch_stringency_index, fetch_holidays


@pytest.mark.order(get_order_number("test_data_collection"))
def test_data_collection():
    weather_data_df = fetch_weather(location_name="Spain", start_date="2020-01-01", end_date="2023-01-01")
    stringency_index_df = fetch_stringency_index('Spain')
    holidays_df = fetch_holidays(years=[2020, 2021, 2022, 2023], country_code="ES")

    assert not weather_data_df.empty, "The weather DataFrame is empty"
    assert weather_data_df.index.name == "Date", "The index name is not set to 'Date' in weather DataFrame"

    assert not stringency_index_df.empty, "The covid DataFrame is empty"
    assert stringency_index_df.index.name == "Date", "The index name is not set to 'Date' in covid DataFrame"

    assert not holidays_df.empty, "The holidays DataFrame is empty"
    assert holidays_df.index.name == "Date", "The index name is not set to 'Date' in holidays DataFrame"  


