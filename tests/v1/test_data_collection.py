import pytest
from tests.v1.conftest import get_order_number
from timepulse.data.data_collection import fetch_stringency_index, fetch_holidays


@pytest.mark.order(get_order_number("test_data_collection"))
def test_data_collection():
    daily_stringency_index_df = fetch_stringency_index("Spain", period="D")
    daily_holidays_df = fetch_holidays(years=[2020, 2021, 2022, 2023], country_code="ES", period="D")
    monthly_stringency_index_df = fetch_stringency_index("Spain", period="M")
    monthly_holidays_df = fetch_holidays(years=[2020, 2021, 2022, 2023], country_code="ES", period="M")

    assert not daily_stringency_index_df.empty, "The daily covid DataFrame is empty"
    assert (
        daily_stringency_index_df.index.name == "Date"
    ), "The index name is not set to 'Date' in daily covid DataFrame"

    assert not daily_holidays_df.empty, "The daily holidays DataFrame is empty"
    assert daily_holidays_df.index.name == "Date", "The index name is not set to 'Date' in daily holidays DataFrame"

    assert not monthly_stringency_index_df.empty, "The monthly covid DataFrame is empty"
    assert (
        monthly_stringency_index_df.index.name == "Date"
    ), "The index name is not set to 'Date' in monthly covid DataFrame"

    assert not monthly_holidays_df.empty, "The monthly holidays DataFrame is empty"
    assert monthly_holidays_df.index.name == "Date", "The index name is not set to 'Date' in monthly holidays DataFrame"

    assert (
        daily_stringency_index_df.shape[0] > monthly_stringency_index_df.shape[0]
    ), "Daily stringency index data should have more entries than monthly data."
    assert (
        daily_holidays_df.shape[0] > monthly_holidays_df.shape[0]
    ), "Daily holidays data should have more entries than monthly data."

    assert (
        daily_stringency_index_df.shape[1] == monthly_stringency_index_df.shape[1]
    ), "Both daily and monthly stringency index data should have the same number of columns."
    assert (
        daily_holidays_df.shape[1] == monthly_holidays_df.shape[1]
    ), "Both daily and monthly holidays data should have the same number of columns."
