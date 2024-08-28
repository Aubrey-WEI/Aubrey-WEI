import yfinance as yf
import pandas as pd
import numpy as np
import csv

company_tickers_annual = 'annual_tickers_2013_2023.csv'
df = pd.read_csv(company_tickers_annual)

#extracts stock symbols in 2013
tickers_2015 = df['2015'].dropna().astype(str)
tickers_2015_list = tickers_2015.tolist()  # 转换为列表

data_annual = []
cov_matrix = []
average_monthly_returns_lists = []
#download stock data in 2013
for ticker in tickers_2015_list:
    data = yf.download(ticker, start='2015-01-01', end = '2015-12-31', interval='1mo')
    #save Adjusted close price
    data_Adj_close = data['Adj Close']

    #calculate monthly return
    monthly_returns_2015 = data_Adj_close.pct_change().dropna()
    monthly_returns_2015 = monthly_returns_2015.fillna(monthly_returns_2015.mean())

    #calculate average monthly return
    average_monthly_returns_2015 = monthly_returns_2015.mean()
    # Handle NaN values (if any) - Replace NaN with 0.01
    if pd.isnull(average_monthly_returns_2015):
        average_monthly_returns_2015 = 0.01
        average_monthly_returns_lists.append(average_monthly_returns_2015)
    data_annual.append(monthly_returns_2015)
    cov_matrix.append(data_annual)


#calculate covariance matrix
# 将每个 Series 合并为一个 DataFrame
# 确保所有 Series 长度相同
df_cov_matrix = pd.DataFrame(data_annual).T
df_cov_matrix.columns = tickers_2015_list

covariance_matrix_2015 = df_cov_matrix.cov()
#print(covariance_matrix_2013)

'''def portfolio_std_dev(weights, cov_matrix):
    """
    Calculate the standard deviation (volatility) of a portfolio given weights and covariance matrix.

    Parameters:
    weights (list or np.array): Array-like object containing weights of assets in the portfolio.
    cov_matrix (pd.DataFrame): Covariance matrix of returns of assets in the portfolio.

    Returns:
    float: Portfolio standard deviation.
    """
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_std_dev = np.sqrt(portfolio_variance)
    return portfolio_std_dev
# Example weights (should sum up to 1)
weights = np.array([0.3, 0.4, 0.3])

# Calculate portfolio standard deviation
portfolio_volatility = portfolio_std_dev(weights, covariance_matrix)

print("Portfolio Standard Deviation (Volatility):", portfolio_volatility)'''

# 保存结果到CSV文件
average_monthly_returns_2015_lists = 'average_monthly_returns_2015.csv'
# Write the data to a CSV file
with open(average_monthly_returns_2015_lists, 'w', newline='') as file:
    writer = csv.writer(file)
    for value in data_annual:
        writer.writerow([value])  # Write each value on a new line

print(f"Data has been written to {average_monthly_returns_2015_lists}")

# 保存协方差矩阵到 Excel
covariance_matrix_2015.to_excel('covariance_matrix_2015.xlsx', sheet_name='Covariance Matrix')


