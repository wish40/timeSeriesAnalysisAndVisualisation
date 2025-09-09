# Import required libraries
import pandas as pd
import numpy as np
from arch import arch_model

# -----------------------
# Step 1: Load the dataset
# -----------------------
data = pd.read_csv("./Reliance.csv")

# Convert date column to datetime
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Ensure 'close' column is 1D
data['close'] = data['close'].astype(float).values.ravel()

# -----------------------
# Step 2: Calculate SMA-20
# -----------------------
# Simple Moving Average of last 20 periods
data['SMA_20'] = data['close'].rolling(window=20).mean()

# -----------------------
# Step 3: Calculate RSI (14-period)
# -----------------------
def calculate_rsi(series, period=14):
    delta = series.diff()

    # Separate gains and losses
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    # Calculate rolling averages
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_rsi(data['close'], period=14)

# -----------------------
# Step 4: Calculate MACD and MACD Signal
# -----------------------
def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

# MACD line = 12-day EMA - 26-day EMA
data['EMA_12'] = calculate_ema(data['close'], 12)
data['EMA_26'] = calculate_ema(data['close'], 26)
data['MACD'] = data['EMA_12'] - data['EMA_26']

# MACD Signal line = 9-day EMA of MACD
data['MACD_signal'] = calculate_ema(data['MACD'], 9)

# -----------------------
# Step 5: Bollinger Bands (20-day, 2 Std Dev)
# -----------------------
rolling_mean = data['close'].rolling(window=20).mean()
rolling_std = data['close'].rolling(window=20).std()

data['BB_high'] = rolling_mean + (rolling_std * 2)
data['BB_low'] = rolling_mean - (rolling_std * 2)

# -----------------------
# Step 6: Daily Returns
# -----------------------
data['returns'] = data['close'].pct_change()

# -----------------------
# Step 7: GARCH(1,1) Volatility Forecast
# -----------------------
returns = data['returns'].dropna() * 100  # Convert to percentage for GARCH

# Build and fit GARCH model
model = arch_model(returns, vol='GARCH', p=1, q=1)
garch_fit = model.fit(disp="off")

# Forecast volatility for next day
vol_forecast = garch_fit.forecast(horizon=1)
data['volatility_forecast'] = np.nan
data.loc[returns.index[-1], 'volatility_forecast'] = np.sqrt(vol_forecast.variance.values[-1, 0])

# -----------------------
# Step 8: Clean Final Data
# -----------------------
data.drop(['EMA_12', 'EMA_26'], axis=1, inplace=True)
data.reset_index(inplace=True)

# Save processed data
data.to_csv("processed_stock_data.csv", index=False)

print("Indicators and GARCH volatility forecast successfully calculated and saved as 'processed_stock_data.csv'")
