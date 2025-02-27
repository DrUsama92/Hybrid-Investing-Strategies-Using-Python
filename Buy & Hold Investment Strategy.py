import pandas as pd
import numpy as np

# Load the DataFrame with the first row as header
df = pd.read_excel('C:/Users/PMLS/Documents/Daily NAV data.xlsx', header=0)
# Strip column names to remove any extra spaces & Drop rows with any NaN values
df.columns = df.columns.str.strip()
df = df.dropna(axis=0, how='any')

# Separate fund information and returns
fund_info_columns = ['Name', 'Ticker', 'SecId', 'ISIN', 'Inception Date']
fund_info = df[fund_info_columns]
returns = df.drop(columns=fund_info_columns)

# Initialize an empty DataFrame to store the percentage changes
percentage_change = pd.DataFrame(index=returns.index)
# Loop through columns and calculate percentage change
for i in range(1, returns.shape[1]):
    prev_col = returns.iloc[:, i - 1]
    curr_col = returns.iloc[:, i]
    # percentage change formula
    percentage_change[f'Pct_Change_{i}'] = (curr_col - prev_col) / prev_col

# Join back with fund_info
result = pd.concat([fund_info, percentage_change], axis=1)
# Calculate percentage change
result.head()

#  In our data percentage_change in columns and funds are rows, Transpose the DataFrame to make funds columns for easier calculation
percentage_change = percentage_change.T
# Initialize lists to store metrics
metrics = {'Name': [], 'Ticker': [], 'SecId': [], 'ISIN': [], 'Inception Date': [], 'Cumulative Return': [],
           'Annualized Return': [], 'Annualized Volatility': [], 'Sharpe Ratio': []}
# 'percentage_change' is the DataFrame with percentage change data
for fund in percentage_change.columns:
    # cumulative returns
    cumulative_returns = (1 + percentage_change[fund]).cumprod() - 1
    final_cumulative_return = cumulative_returns.iloc[-1]
    # Annualized Return
    annualized_return = ((1 + final_cumulative_return) ** (365 / len(percentage_change[fund]))) - 1
    # Annualized Volatility
    annualized_volatility = percentage_change[fund].std() * np.sqrt(365)
    # Sharpe Ratio (assuming a risk-free rate of 0 for simplicity)
    sharpe_ratio = annualized_return / annualized_volatility
    # Extract fund info
    fund_row = fund_info.loc[fund_info.index[percentage_change.columns.get_loc(fund)]]

    # Append results to metrics
    metrics['Name'].append(fund_row['Name'])
    metrics['Ticker'].append(fund_row['Ticker'])
    metrics['SecId'].append(fund_row['SecId'])
    metrics['ISIN'].append(fund_row['ISIN'])
    metrics['Inception Date'].append(fund_row['Inception Date'])
    metrics['Cumulative Return'].append(final_cumulative_return)
    metrics['Annualized Return'].append(annualized_return)
    metrics['Annualized Volatility'].append(annualized_volatility)
    metrics['Sharpe Ratio'].append(sharpe_ratio)

# DataFrame for the metrics
metrics_df = pd.DataFrame(metrics)
# Apply shortlisting criteria
shortlisted_funds = metrics_df[(metrics_df['Sharpe Ratio'] > 1) | (metrics_df['Cumulative Return'] > 0)]
shortlisted_funds
shortlisted_funds.to_excel('C:/Users/PMLS/Downloads/shortlist_funds.xlsx', index=False)

# SECOND CRITERIA
# Rank funds based on cumulative returns
metrics_df['Rank'] = metrics_df['Cumulative Return'].rank(ascending=False)
# Define the number of top funds to shortlist
top_n = 100
# Select the top N funds with the highest cumulative returns
shortlist_funds = metrics_df.nsmallest(top_n, 'Rank')
shortlist_funds
shortlist_funds.to_excel('C:/Users/PMLS/Downloads/top_funds_shortlist.xlsx', index=False)



