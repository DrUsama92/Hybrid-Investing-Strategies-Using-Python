
import pandas as pd
import numpy as np
# Load data
data = pd.read_excel('C:/Users/PMLS/Documents/ESG data.xlsx')
# Drop rows with missing values
data.dropna(inplace=True)
# Calculate average ESG Managed Risk Score
esg_columns = [col for col in data.columns if 'Portfolio Corporate ESG Managed Risk Score' in col]
data['Average_ESG_Risk_Score'] = data[esg_columns].mean(axis=1)
# Calculate average emissions for each scope
data['Average_Scope1_Emissions'] = data.filter(like='Absolute Carbon Emissions Scope 1').mean(axis=1)
data['Average_Scope2_Emissions'] = data.filter(like='Absolute Carbon Emissions Scope 2').mean(axis=1)
data['Average_Scope3_Emissions'] = data.filter(like='Absolute Carbon Emissions Scope 3').mean(axis=1)
# Calculate average Sustainability Score
data['Average_Sustainability_Score'] = data.filter(like='Historical Corporate Sustainability Score').mean(axis=1)

# Determine median thresholds for shortlisting
mean_scope1_emissions = data['Average_Scope1_Emissions'].mean()
mean_scope2_emissions = data['Average_Scope2_Emissions'].mean()
mean_scope3_emissions = data['Average_Scope3_Emissions'].mean()
mean_esg_risk_score = data['Average_ESG_Risk_Score'].mean()
mean_sustainability_score = data['Average_Sustainability_Score'].mean()

# Define criteria based on median thresholds
data['Meets_Criteria'] = (
    (data['Average_Scope1_Emissions'] <= mean_scope1_emissions) &
    (data['Average_Scope2_Emissions'] <= mean_scope2_emissions) &
    (data['Average_Scope3_Emissions'] <= mean_scope3_emissions) &
    (data['Average_ESG_Risk_Score'] <= mean_esg_risk_score) &
    (data['Average_Sustainability_Score'] >= mean_sustainability_score)
)

# Filter funds that meet the criteria
esg_funds = data[data['Meets_Criteria']]

# Select only the columns to include in the output
columns_to_include = ['Name', 'Ticker', 'SecId', 'ISIN', 'Inception Date',
                      'Average_Scope1_Emissions', 'Average_Scope2_Emissions',
                      'Average_Scope3_Emissions', 'Average_ESG_Risk_Score',
                      'Average_Sustainability_Score']
esg_funds = esg_funds[columns_to_include]
# Save the filtered data to an Excel file
esg_funds.to_excel('C:/Users/PMLS/Documents/Shortlisted_ESG_Funds.xlsx', index=False)

## 2nd selection Criteria
data['Meets_Criteria'] = (((data['Average_Scope1_Emissions'] <= mean_scope1_emissions) |
                           (data['Average_Scope2_Emissions'] <= mean_scope2_emissions) |
                           (data['Average_Scope3_Emissions'] <= mean_scope3_emissions)) &(
                         (data['Average_ESG_Risk_Score'] <= mean_esg_risk_score) &
                         (data['Average_Sustainability_Score'] >= mean_sustainability_score)))

esg_funds = data[data['Meets_Criteria']]

# Select only the columns to include in the output
columns_to_include = ['Name', 'Ticker', 'SecId', 'ISIN', 'Inception Date',
                      'Average_Scope1_Emissions', 'Average_Scope2_Emissions',
                      'Average_Scope3_Emissions', 'Average_ESG_Risk_Score',
                      'Average_Sustainability_Score']

# Create the final DataFrame with the selected columns
esg_funds = esg_funds[columns_to_include]
esg_funds.to_excel('C:/Users/PMLS/Documents/2nd_critera_Shortlisted_ESG_Funds.xlsx', index=False)






