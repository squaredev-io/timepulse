from statsmodels.tsa.stattools import adfuller

def perform_adfuller(df):
    # 2. Remove Trends (Example: First-order differencing)
    differenced_series = df.diff().dropna()

    # 3. Remove Seasonality (if necessary)
    # seasonal_differenced_series = differenced_series.diff(12).dropna()  # Seasonal differencing with a period of 12 (for monthly data)

    # 4. Check for constant variance
    result = adfuller(differenced_series)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values (1%):", result[4]['1%'])
    print("Critical Values (5%):", result[4]['5%'])
    print("Critical Values (10%):", result[4]['10%'])

    # Check if the time series is stationary or not
    if result[1] <= 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is not stationary.")
