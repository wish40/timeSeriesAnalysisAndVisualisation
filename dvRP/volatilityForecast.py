# ==============================
# Step 1: Import Required Libraries
# ==============================
import pandas as pd
import numpy as np
import ta  # Technical Analysis library
from arch import arch_model  # For GARCH model
import matplotlib.pyplot as plt
import warnings

# Suppress convergence warnings from the GARCH model for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================
# Step 2: Load Your Data
# ==============================
try:
    # Load the data from the uploaded CSV file
    df = pd.read_csv('./processed_stock_data.csv', parse_dates=['date'])
    print("Successfully loaded processed_stock_data.csv")
except FileNotFoundError:
    print("Error: 'processed_stock_data.csv' not found. Please ensure the file is in the same directory.")
    exit()

# Ensure the dataframe is sorted by date for accurate calculations
df.sort_values('date', inplace=True)
df.reset_index(drop=True, inplace=True)

# ==============================
# Step 3: Calculate RSI (Relative Strength Index)
# ==============================
# Using the 'ta' library to calculate RSI with a standard 14-day window
rsi_indicator = ta.momentum.RSIIndicator(close=df['close'], window=14)
df['RSI_calculated'] = rsi_indicator.rsi()
print("RSI values have been calculated.")

# ==============================
# Step 4: Calculate GARCH Volatility Forecast
# ==============================
# First, calculate daily returns from the 'close' price
# The first value will be NaN, as there's no previous day to compare.
df['returns_calculated'] = df['close'].pct_change()

# Prepare the returns data for the GARCH model
# We drop the initial NaN value and multiply by 100 for better model stability
returns = df['returns_calculated'].dropna() * 100

if not returns.empty:
    # Define and fit the GARCH(1,1) model
    # This model is commonly used for financial time series volatility
    garch_model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = garch_model.fit(disp='off') # 'disp=off' hides the convergence output

    # Forecast volatility for the next 30 days
    forecast_horizon = 30
    forecast = model_fit.forecast(horizon=forecast_horizon)

    # Extract the forecasted variance for the immediate next day
    next_day_variance = forecast.variance.values[-1, 0]

    # Create a new column for the forecast and add the value to the last row
    df['volatility_forecast_calculated'] = np.nan
    df.loc[df.index[-1], 'volatility_forecast_calculated'] = next_day_variance
    print(f"GARCH volatility forecast for the next day: {next_day_variance:.4f}")

    # ==============================
    # Step 5: Visualize the Volatility Forecast
    # ==============================
    # Create a date range for the forecast period
    last_date = df['date'].iloc[-1]
    forecast_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, forecast_horizon + 1)])
    
    # Plotting standard deviation (sqrt of variance) is more intuitive
    forecasted_std_dev = np.sqrt(forecast.variance.iloc[-1].values)

    plt.figure(figsize=(12, 6))
    plt.plot(forecast_dates, forecasted_std_dev, label='Forecasted Volatility (Std. Dev.)', color='coral', marker='o')
    plt.title(f"{forecast_horizon}-Day Volatility Forecast", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Forecasted Volatility")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('volatility_forecast_plot.png')
    print("Volatility forecast plot saved as 'volatility_forecast_plot.png'")
    # To display the plot when running locally, uncomment the next line
    # plt.show()

else:
    print("Could not calculate volatility forecast because there are no returns data.")

# ==============================
# Step 6: Save the Updated Data
# ==============================
# Save the dataframe with the new columns to a new CSV file
df.to_csv('calculated_stock_data.csv', index=False)
print("\nCalculation complete. The updated data has been saved to 'calculated_stock_data.csv'")
