# ==============================
# Step 1: Import Required Libraries
# ==============================
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta # Import pandas_ta
import matplotlib.pyplot as plt
from arch import arch_model  # GARCH model

# ==============================
# Step 2: Download Stock Data
# ==============================
ticker = "RELIANCE.NS"
# The end date for yfinance is exclusive. Using 2025-01-01 includes data up to 2024-12-31.
data = yf.download(ticker, start="2020-01-01", end="2025-09-01")

# It's good practice to check if the download was successful
if data.empty:
    print(f"No data found for ticker {ticker}. Please check the ticker symbol and date range.")
    exit()

data.reset_index(inplace=True)
data.rename(columns={
    'Date': 'date',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Adj Close': 'adj_close',
    'Volume': 'volume'
}, inplace=True)

# ==============================
# Step 3: Ensure 'close' column is 1D Series and handle missing values
# ==============================
# Convert the 'close' column to a 1D Series explicitly
data['close'] = pd.Series(data['close'].squeeze())

# Check for and handle missing values in the 'close' column
try:
    if data['close'].isnull().any():
        print("Warning: Missing values found in 'close' column. Filling NaN with forward fill.")
        data['close'] = data['close'].fillna(method='ffill')
        # If there are still NaNs at the beginning, fill with backward fill
        data['close'] = data['close'].fillna(method='bfill')
except ValueError as e:
    print(f"Error checking for null values: {e}")
    print("Proceeding with filling NaN values anyway.")
    data['close'] = data['close'].fillna(method='ffill')
    data['close'] = data['close'].fillna(method='bfill')


print(f"Type of 'close' column: {type(data['close'])}")
print(f"Shape of 'close' column: {data['close'].shape}")


# ==============================
# Step 4: Create Technical Indicators using pandas_ta
# ==============================
data['SMA_20'] = ta.sma(data['close'], length=20)
data['RSI'] = ta.rsi(data['close'], length=14)
macd_results = ta.macd(data['close'], fast=12, slow=26, signal=9) # Default parameters
# Check if macd_results is not None before accessing
if macd_results is not None:
    data['MACD'] = macd_results['MACD_12_26_9']
    data['MACD_signal'] = macd_results['MACDh_12_26_9'] # Note: pandas_ta provides MACD Histogram as MACDh
else:
    print("Warning: MACD calculation failed. Skipping MACD indicators.")
    data['MACD'] = np.nan
    data['MACD_signal'] = np.nan

# Using the default Bollinger Bands parameters
bollinger_results = ta.bbands(data['close'], length=20, std=2)
# Check if bollinger_results is not None before accessing
if bollinger_results is not None:
    data['BB_high'] = bollinger_results['BBU_20_2.0']
    data['BB_low'] = bollinger_results['BBL_20_2.0']
else:
    print("Warning: Bollinger Bands calculation failed. Skipping Bollinger Bands indicators.")
    data['BB_high'] = np.nan
    data['BB_low'] = np.nan


# ==============================
# Step 5: GARCH Model Forecasting
# ==============================
# Calculate percentage returns. The first value will be NaN.
data['returns'] = data['close'].pct_change()

# Prepare returns for the GARCH model: drop NaN and scale by 100 for better model convergence.
returns = data['returns'].dropna() * 100

# Fit the GARCH(1,1) model.
model = arch_model(returns, vol='Garch', p=1, q=1)
model_fit = model.fit(disp='off')

# Forecast volatility for the next 30 days.
forecast_horizon = 30
forecast = model_fit.forecast(horizon=forecast_horizon)

# Extract the forecasted variance for the next day to use in KPIs.
next_day_forecast = forecast.variance.values[-1, 0]

# Create a placeholder column and assign the forecast to the last row for the CSV export.
data['volatility_forecast'] = np.nan
data.loc[data.index[-1], 'volatility_forecast'] = next_day_forecast

# ==============================
# Step 6: Generate KPIs
# ==============================
data['daily_return'] = data['close'].pct_change()
# Fill the first NaN value in cumulative_return with 1 (representing the starting point).
data['cumulative_return'] = (1 + data['daily_return']).cumprod().fillna(1)

KPI_1 = data['close'].iloc[-1].item() # Extract scalar value
KPI_2 = data['daily_return'].mean() * 100
KPI_3 = data['volatility_forecast'].iloc[-1].item() # Extract scalar value
KPI_4 = data['cumulative_return'].iloc[-1].item() # Extract scalar value

print("\n--- Key Performance Indicators ---")
print(f"KPI 1 (Latest Close Price): ${KPI_1:.2f}")
print(f"KPI 2 (Average Daily Return %): {KPI_2:.4f}%")
print(f"KPI 3 (Forecasted Volatility, next day): {KPI_3:.4f}")
print(f"KPI 4 (Cumulative Return): {KPI_4:.2f}")
print("--------------------------------\n")

# ==============================
# Step 7: Prepare Data for Power BI
# ==============================
export_columns = [
    'date', 'open', 'high', 'low', 'close', 'volume',
    'SMA_20', 'RSI', 'MACD', 'MACD_signal',
    'BB_high', 'BB_low', 'returns', 'volatility_forecast',
    'daily_return', 'cumulative_return'
]
final_data = data[export_columns]
final_data.to_csv("financial_dashboard_data.csv", index=False)
print("Data successfully exported to financial_dashboard_data.csv")

# ==============================
# Step 8: Visualize GARCH Volatility (CORRECTED & IMPROVED)
# ==============================
# Plot 1: Daily Returns
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['returns'], label='Daily Returns', color='royalblue', alpha=0.8)
plt.title(f"{ticker} Daily Returns Over Time", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Return")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Plot 2: GARCH Forecasted Volatility
# Create a date range for the forecast horizon.
last_date = data['date'].iloc[-1]
forecast_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, forecast_horizon + 1)])
forecasted_values = np.sqrt(forecast.variance.iloc[-1].values) # Plotting std dev (sqrt of variance) is more intuitive

plt.figure(figsize=(12, 6))
plt.plot(forecast_dates, forecasted_values, label='Forecasted Volatility (Std. Dev.)', color='red', marker='o')
plt.title(f"{ticker} GARCH Model {forecast_horizon}-Day Volatility Forecast", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Forecasted Volatility (Standard Deviation)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()