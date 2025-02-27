import pandas as pd
import numpy as np

# Load daily NAV data
data = pd.read_excel('C:/Users/123/Desktop/Project work/daily nav data 2.xlsx')

# Strip column names to remove any extra spaces & drop rows with any missing values
data.columns = data.columns.str.strip()
data = data.dropna(axis=0, how='any')

# Initialize an empty DataFrame to store the percentage changes
monthly_pct_change = pd.DataFrame(index=monthly_returns.index)
# Loop through columns and calculate percentage change
for i in range(1, monthly_returns.shape[1]):
    prev_col = monthly_returns.iloc[:, i - 1]
    curr_col = monthly_returns.iloc[:, i]
    # percentage change formula
    monthly_pct_change[f'mon_Pct_Change_{i}'] = (curr_col - prev_col) / prev_col
# Join back with fund_info
result = pd.concat([fund_info, monthly_pct_change], axis=1)
# Calculate percentage change
result.head()

# Initialize a DataFrame to store the market_neutral_returns
mkt_neutral_returns = pd.DataFrame(index=monthly_pct_change.columns)

# Initialize an empty list to store market-neutral returns with their respective month
market_neutral_returns_list = []

# Variable to track the maximum market-neutral return and its corresponding month
max_market_neutral_return = float('-inf')
max_month = None
max_long_assets = None
max_short_assets = None

for month in monthly_pct_change.columns:
    print(f"Processing: {month}")

    # Calculate the mean return for each fund in the month
    mean_returns = monthly_pct_change[month]

    # Calculate the benchmark (mean of all fund means for the month)
    benchmark = mean_returns.mean()

    # Identify long (above benchmark) and short (below or equal to benchmark) positions
    long_assets = mean_returns[mean_returns > benchmark].sort_values(ascending=False)
    short_assets = mean_returns[mean_returns <= benchmark].sort_values(ascending=True)

    # Ensure equal number of long and short positions
    min_len = 150
    long_assets = long_assets.head(150)
    short_assets = short_assets.head(150)

    # Calculate the market-neutral portfolio return for the month
    long_return = long_assets.mean()
    short_return = short_assets.mean()
    mkt_neutral_return = long_return - short_return

    # Save the result in mkt_neutral_returns DataFrame
    mkt_neutral_returns.loc[month, 'Market Neutral Return'] = mkt_neutral_return

    # Append the month and the market-neutral return to the list
    market_neutral_returns_list.append((month, mkt_neutral_return))

    # Track the maximum market-neutral return and the respective month and assets
    if mkt_neutral_return > max_market_neutral_return:
        max_market_neutral_return = mkt_neutral_return
        max_month = month
        max_long_assets = long_assets
        max_short_assets = short_assets

# Print the market-neutral returns list after the loop
print("\nMarket Neutral Returns List:")
print(market_neutral_returns_list)

# Print the maximum market-neutral return and its corresponding long and short assets
print(f"\nMaximum Market Neutral Return: {max_market_neutral_return}")
print(f"Month with Maximum Return: {max_month}")
print("\nLong Assets for the month with maximum return:")
print(max_long_assets)
print("\nShort Assets for the month with maximum return:")
print(max_short_assets)

# Extract fund info for long and short assets for the month with maximum return
long_asset_funds = fund_info.loc[max_long_assets.index]
short_asset_funds = fund_info.loc[max_short_assets.index]

# Using .copy() to avoid SettingWithCopyWarning
long_asset_funds = long_asset_funds.copy()
short_asset_funds = short_asset_funds.copy()

# Label the long and short assets
long_asset_funds['Position'] = 'Long'
short_asset_funds['Position'] = 'Short'

# Combine both long and short funds into a single DataFrame
shortlisted_funds = pd.concat([long_asset_funds, short_asset_funds])
shortlisted_funds

# Save the shortlisted funds in an excel file
shortlisted_funds.to_excel('C:/Users/123/Downloads/shortlisted_funds_1.xlsx', index=False)