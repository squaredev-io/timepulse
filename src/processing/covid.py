import pandas as pd
import re

def fetch_stringency_index(country):
    """
    Fetches and preprocesses monthly stringency index data for the specified country.

    Parameters
    ----------
    country : str
        Country name for which the stringency index data is fetched.

    Returns
    -------
    pd.DataFrame
        Monthly stringency index data with a calculated mean for 'stringency_index' and mode for 'stringency_category'.

    Example
    -------
    monthly_strigency_index_df = fetch_stringency_index('Spain')
    """
    stringency_index_avg_url = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/timeseries/stringency_index_avg.csv'
    spain_strigency_index_df = pd.read_csv(stringency_index_avg_url)
    spain_strigency_index_df = spain_strigency_index_df[spain_strigency_index_df['country_name']==country]

    # Define a regular expression pattern to match the date format
    date_pattern = r'(\d{2})([A-Za-z]{3})(\d{4})'
    date_columns = [col for col in spain_strigency_index_df.columns if re.fullmatch(date_pattern, col)]

    spain_strigency_index_df = spain_strigency_index_df[date_columns]
    spain_strigency_index_df = spain_strigency_index_df.fillna(0)

    # Melt the DataFrame to unpivot it
    spain_strigency_index_df = pd.melt(spain_strigency_index_df, id_vars=None, var_name='Date', value_name='stringency_index')

    # Convert the 'Date' column to a datetime format in 'yyyy-mm-dd' format
    spain_strigency_index_df['Date'] = pd.to_datetime(spain_strigency_index_df['Date'], format='%d%b%Y').dt.strftime('%Y-%m-%d')
    spain_strigency_index_df['Date'] = pd.to_datetime(spain_strigency_index_df['Date'])

    # Optionally, sort the DataFrame by date
    spain_strigency_index_df = spain_strigency_index_df.sort_values(by='Date')
    spain_strigency_index_df = spain_strigency_index_df.set_index('Date')

    # Define bin edges and labels
    bin_edges = [-1, 33., 66., 110.]  # Example boundaries for low, medium, high
    bin_labels = [0, 1, 2]

    # Create the 'stringency_category' column
    spain_strigency_index_df['stringency_category'] = pd.cut(spain_strigency_index_df['stringency_index'], bins=bin_edges, labels=bin_labels)

    # Display the DataFrame with the new 'stringency_category' column
    spain_strigency_index_df['stringency_category'].value_counts()

    # Resample to monthly and calculate mean for 'stringency_index' and mode for 'stringency_category'
    monthly_strigency_index_df = spain_strigency_index_df.resample('M').agg({
        'stringency_index': 'mean',
        'stringency_category': lambda x: x.mode().iloc[0]
    })

    return monthly_strigency_index_df
