import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load the data
data = pd.read_excel('C:/Users/PMLS/Documents/Funds_data.xlsx')
# Load the risk free rate data
df = pd.read_excel('C:/Users/PMLS/Documents/risk free rate data.xlsx')
# Clean and preprocess data
data.rename(columns={'Yearly Return Year2021 Base Currency': 'Return 2021',
                     'Yearly Return Year2022 Base Currency': 'Return 2022',
                     'Yearly Return Year2023 Base Currency': 'Return 2023'}, inplace=True)
data.columns = data.columns.str.strip()
data = data.drop(columns=['Return 2021'], axis=1)
data = data.dropna(axis=0, how='any')
data['Return 2022'] = pd.to_numeric(data['Return 2022'], errors='coerce')
data['Return 2023'] = pd.to_numeric(data['Return 2023'], errors='coerce')
# Calculate expected returns (mean of 2022 and 2023 returns)
data['Expected Return'] = data[['Return 2022', 'Return 2023']].mean(axis=1)
data['Std_Dev'] = data[['Return 2022', 'Return 2023']].apply(lambda x: x.std(), axis=1)

# Transpose the DataFrame for easier access to fund details and returns
data = data.set_index('Name').T
data

# Calculate mean returns and covariance matrix
mean_returns = data.loc['Expected Return']
cov_matrix = data.loc[['Return 2022', 'Return 2023']].cov()
cov_matrix

# Extract year from date
df['Year'] = df['Date'].dt.year
# Calculate annual risk-free rate as the average of monthly rates for each year
risk_free_rate = df.groupby('Year')['Risk free rate'].mean().reset_index()
# Filter for 2022 and 2023
df_filtered = risk_free_rate[risk_free_rate['Year'].isin([2022, 2023])]
# Calculate the mean risk-free rate for 2022 and 2023
mean_risk_free_rate = df_filtered['Risk free rate'].mean()
mean_risk_free_rate

# Simulate portfolios
num_assets = len(mean_returns)
num_portfolios = 10000
results = np.zeros((num_portfolios, 3))  # [Return, StdDev, Sharpe]
np.random.seed(42)  # For reproducibility
for i in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)  # Normalize weights

    portfolio_return = np.dot(weights, mean_returns)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - mean_risk_free_rate) / portfolio_std_dev

    results[i, 0] = portfolio_return
    results[i, 1] = portfolio_std_dev
    results[i, 2] = sharpe_ratio

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=['Return', 'StdDev', 'Sharpe'])

# Find the portfolio with maximum Sharpe Ratio
max_sharpe_idx = results_df['Sharpe'].idxmax()
max_sharpe_portfolio = results_df.iloc[max_sharpe_idx]

# Find the portfolio with minimum volatility
min_volatility_idx = results_df['StdDev'].idxmin()
min_volatility_portfolio = results_df.iloc[min_volatility_idx]

print("Maximum Sharpe Ratio Portfolio:")
print(f"Return: {max_sharpe_portfolio['Return']:.2f}")
print(f"Standard Deviation: {max_sharpe_portfolio['StdDev']:.2f}")
print(f"Sharpe Ratio: {max_sharpe_portfolio['Sharpe']:.2f}")

print("\nMinimum Volatility Portfolio:")
print(f"Return: {min_volatility_portfolio['Return']:.2f}")
print(f"Standard Deviation: {min_volatility_portfolio['StdDev']:.2f}")
print(f"Sharpe Ratio: {min_volatility_portfolio['Sharpe']:.2f}")

# Plot Efficient Frontier
from scipy.spatial import ConvexHull

# Extract data for convex hull calculation
points = np.array(results_df[['StdDev', 'Return']])
# Compute the convex hull & Plot the convex hull
hull = ConvexHull(points)
hull_points = points[hull.vertices]

plt.figure(figsize=(10, 6))
plt.scatter(results_df['StdDev'], results_df['Return'], c=results_df['Sharpe'], cmap='viridis', marker='o')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Standard Deviation (Risk)')
plt.ylabel('Return')
plt.title('Efficient Frontier')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k--', lw=2)  # 'k--' for black dashed line
plt.show()


# Optimize Portfolio Weights
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std_dev


def negative_sharpe_ratio(weights, mean_returns, cov_matrix, mean_risk_free_rate):
    p_return, p_std_dev = portfolio_performance(weights, mean_returns, cov_matrix)
    return - (p_return - mean_risk_free_rate) / p_std_dev


init_weights = num_assets * [1. / num_assets]
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for asset in range(num_assets))

opt_results = minimize(negative_sharpe_ratio, init_weights, args=(mean_returns, cov_matrix, mean_risk_free_rate),
                       method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = opt_results.x

# Shortlist Funds
weights_df = pd.DataFrame({'Fund': data.columns, 'Optimal Weight': optimal_weights})
threshold = np.percentile(optimal_weights, 50)
shortlisted_funds = weights_df[weights_df['Optimal Weight'] > threshold]

# Merge the shortlisted funds with fund details
fund_details = data.T[['Ticker', 'SecId', 'ISIN', 'Inception Date']]
shortlisted_funds = shortlisted_funds.merge(fund_details, left_on='Fund', right_index=True)
shortlisted_funds.to_excel('C:/Users/PMLS/Downloads/funds_after_shortlisting.xlsx', index=False)

