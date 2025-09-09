import yfinance as yf
import pandas as pd

# Set pandas to display all columns
pd.set_option('display.max_columns', None)

# Define the ticker symbol for Nifty 50
nifty_ticker = "^NSEI"

# Download the historical data
# You can specify start and end dates, or use a period like '1y', '5y', 'max'
try:
    data = yf.download(nifty_ticker, start="2020-01-01")

    # Check if the dataframe is empty
    if data.empty:
        print(f"No data found for ticker {nifty_ticker}. It might be delisted or an incorrect ticker.")
    else:
        # Display the last 5 rows of the downloaded data
        print(f"Successfully downloaded data for Nifty 50 ({nifty_ticker})")
        print("---------------------------------------------------------")
        print("Last 5 trading days:")
        print(data.tail())

        # (Optional) Save the data to a CSV file
        data.to_csv("nifty50_data.csv")
        print("\nData saved to nifty50_data.csv")

except Exception as e:
    print(f"An error occurred: {e}")