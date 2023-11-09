import pandas as pd


def handle_missing_values(df):
    processed_df = df.copy()
    processed_df['value'] = processed_df['value'].bfill()

    processed_df['date'] = pd.to_datetime(processed_df[['year', 'month']].assign(day=1)) + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    processed_df['date'] = processed_df['date'].dt.strftime('%Y-%m-%d')
    processed_df.set_index('date', inplace=True)
    return processed_df