import pandas as pd
import re  # Import regular expressions for date extraction
import warnings

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Step 1: Load the data
df = pd.read_excel('C:/Users/PMLS/Downloads/data for strategy.xlsx')

# Step 2: Extract P/E, P/B, and Dividend columns & Calculate averages
pe_columns = df.filter(like='P/E').columns
df['Avg_P/E'] = df[pe_columns].mean(axis=1)
pb_columns = df.filter(like='P/B').columns
df['Avg_P/B'] = df[pb_columns].mean(axis=1)
pcf_columns = df.filter(like='P/C').columns
df['Avg_P/CF'] = df[pcf_columns].mean(axis=1)

# Step 3: Drop the original columns of P/E, P/B
df = df.drop(columns=pe_columns)
df = df.drop(columns=pb_columns)
df = df.drop(columns=pcf_columns)
# Step 4: Drop rows with missing values
df = df.dropna()
df

# Create a DataFrame with NAV data only
nav_columns = [col for col in df.columns if 'Raw Return (NAV)' in col]
nav_df = df[nav_columns]

# Step 5: Extract the fund details (columns that are not NAV)
fund_details_columns = [col for col in df.columns if col not in nav_columns]
fund_details_df = df[fund_details_columns]  # Create a DataFrame with fund details only


def extract_date_from_column(col_name):
    match = re.search(r'\d{4}-\d{2}-\d{2}', col_name)
    return match.group(0) if match else None


nav_df.columns = [extract_date_from_column(col) for col in nav_df.columns]
nav_df.columns = pd.to_datetime(nav_df.columns)
monthly_nav_df = nav_df.resample('M', axis=1).last()
monthly_nav_df
# Step 6: Calculate NAV change over the last 6 months
six_months_ago = monthly_nav_df.iloc[:, -6]  # previous 6th month NAV
last_month_nav = monthly_nav_df.iloc[:, -1]  # NAV last month
monthly_nav_df['6_Month_Change'] = ((last_month_nav - six_months_ago) / six_months_ago) * 100

# Step 7: Add the percentage change to the original DataFrame
results_df = pd.concat([fund_details_df, monthly_nav_df[['6_Month_Change']]], axis=1)

# Step 8: Rank each metric
results_df['Momentum Score Rank'] = results_df['6_Month_Change'].rank(method='dense', ascending=True)
results_df['Momentum Score Rank'] = ((results_df['Momentum Score Rank'] / len(results_df)) * 100)  # Scale to 1-100


def rank_metrics(df):
    metrics = ['Avg_P/E', 'Avg_P/B', 'Avg_P/CF']

    for metric in metrics:
        df[f'{metric} Rank'] = df[metric].rank(method='dense', ascending=True)
        # Scale to 1-100
        df[f'{metric} Rank'] = ((df[f'{metric} Rank'] / len(df)) * 100)


rank_metrics(results_df)


def calculate_composite_score(df):
    # Calculate composite score as the average of the ranks
    df['Composite Score'] = df[['Avg_P/E Rank', 'Avg_P/B Rank', 'Avg_P/CF Rank']].mean(axis=1)

    return df


results_df = calculate_composite_score(results_df)
# Step 9: Calculate Final Score (Composite Score + Momentum Score)
results_df['Trending value score'] = results_df['Composite Score'] + results_df['Momentum Score Rank']
# Step 10: Shortlist funds
# Set a threshold for low composite score and high momentum score
low_composite_threshold = results_df['Composite Score'].quantile(0.4)
high_momentum_threshold = results_df['Momentum Score Rank'].quantile(0.6)
# Filter the results based on the thresholds
shortlisted_funds = results_df[
    (results_df['Composite Score'] <= low_composite_threshold) &
    (results_df['Momentum Score Rank'] >= high_momentum_threshold)]
shortlisted_funds
shortlisted_funds.to_excel('C:/Users/PMLS/Documents/Undervalued_HighMomentum_Funds.xlsx', index=False)







