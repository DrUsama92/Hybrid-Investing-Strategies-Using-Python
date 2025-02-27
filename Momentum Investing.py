import pandas as pd

# Load the data from the uploaded Excel file
file_path = 'd:/Momentum Investing.xlsx'
df = pd.read_excel(file_path)

# Ensure the column names for Average Gain and Average Loss are correct
average_gain_column = 'Average Gain'
average_loss_column = 'Average Loss'

# Calculate Relative Strength (RS)
df['RS'] = df[average_gain_column] / df[average_loss_column]

# Calculate RSI
df['RSI'] = 100 - (100 / (1 + df['RS']))

# Criteria for high momentum funds
momentum_factor_column = 'Momentum Factor - Multi-Asset Model'
profile_momentum_column = 'Factor Profile - Momentum'

# Specify columns of interest
columns_of_interest = ['Name', 'SecId', 'ISIN', 'Inception Date', profile_momentum_column,
                       momentum_factor_column, average_gain_column, average_loss_column, 'RSI']

# Drop rows with missing values in the columns of interest
df.dropna(subset=columns_of_interest, inplace=True)

# Calculate average values for thresholds
average_momentum_factor = df[momentum_factor_column].mean()
average_profile_momentum = df[profile_momentum_column].mean()
rsi_threshold = 70  # Assuming RSI > 70 indicates high momentum

# Define thresholds for high momentum criteria
momentum_factor_threshold = average_momentum_factor
profile_momentum_threshold = average_profile_momentum

# Filter high momentum funds - Conservative approach (Approach#01)
high_momentum_funds_01 = df[
    (df[momentum_factor_column] > momentum_factor_threshold) &
    (df[profile_momentum_column] > profile_momentum_threshold) &
    (df['RSI'] > rsi_threshold)
]

# Filter high momentum funds - Liberal approach (Approach#02)
high_momentum_funds_02 = df[
    (df[momentum_factor_column] > momentum_factor_threshold) |
    (df[profile_momentum_column] > profile_momentum_threshold) |
    (df['RSI'] > rsi_threshold)
]

# Select only the specified columns for export
columns_to_export = ['Name', 'SecId', 'ISIN', 'Inception Date', profile_momentum_column,
                     momentum_factor_column, average_gain_column, average_loss_column, 'RSI']

high_momentum_funds_01_selected = high_momentum_funds_01[columns_to_export]
high_momentum_funds_02_selected = high_momentum_funds_02[columns_to_export]

# Export the filtered high momentum funds to Excel files
output_file_path_01 = 'd:/High_Momentum_Funds_Conservative.xlsx'
output_file_path_02 = 'd:/High_Momentum_Funds_Liberal.xlsx'
high_momentum_funds_01_selected.to_excel(output_file_path_01, index=False, sheet_name='Conservative Approach')
high_momentum_funds_02_selected.to_excel(output_file_path_02, index=False, sheet_name='Liberal Approach')
