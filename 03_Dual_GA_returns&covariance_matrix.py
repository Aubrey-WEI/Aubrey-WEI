import yfinance as yf
import pandas as pd
import numpy as np
import csv

company_tickers_annual = 'Dual_GA_2014.xlsx'
df = pd.read_excel(company_tickers_annual)

#extracts stock symbols in 2013
tickers_2014 = df['tickers'].dropna().astype(str)
tickers_2014_list = tickers_2014.tolist()  # 转换为列表

data_annual = []
cov_matrix = []
average_monthly_returns_lists = []
stock_std_dev = []
#download stock data in 2013
for ticker in tickers_2014_list:
    data = yf.download(ticker, start='2014-01-01', end = '2014-12-31', interval='1mo')
    #save Adjusted close price
    data_Adj_close = data['Adj Close']

    #calculate monthly return
    monthly_returns_2014 = data_Adj_close.pct_change().dropna()
    monthly_returns_2014 = monthly_returns_2014.fillna(monthly_returns_2014.mean())
    # calculate individual stock's deviation in order to calculate Diversification Ratio
    std_dev = monthly_returns_2014.std()
    stock_std_dev.append(std_dev)
    #calculate average monthly return
    average_monthly_returns_2014 = monthly_returns_2014.mean()
    # Handle NaN values (if any) - Replace NaN with 0.01
    if pd.isnull(average_monthly_returns_2014):
        average_monthly_returns_2014 = 0.01
    average_monthly_returns_lists.append(average_monthly_returns_2014)
    data_annual.append(monthly_returns_2014)
    cov_matrix.append(data_annual)
#print(average_monthly_returns_2013)
#calculate covariance matrix
# 将每个 Series 合并为一个 DataFrame
# 确保所有 Series 长度相同
df_cov_matrix = pd.DataFrame(data_annual).T
df_cov_matrix.columns = tickers_2014_list

covariance_matrix_2014 = df_cov_matrix.cov()
#print(covariance_matrix_2013)
#print(stock_std_dev)
df = pd.DataFrame(stock_std_dev, columns=["Stock Deviation"])
df.to_excel('individual_std_dev.xlsx', index=False)

# 保存结果到CSV文件
average_monthly_returns_lists = 'dual_average_monthly_returns_2014.csv'
# Write the data to a CSV file
with open(average_monthly_returns_lists, 'w', newline='') as file:
    writer = csv.writer(file)
    for value in data_annual:
        writer.writerow([value])  # Write each value on a new line

print(f"Data has been written to {average_monthly_returns_lists}")

# 保存协方差矩阵到 Excel
covariance_matrix_2014.to_excel('Dual_GA_covariance_matrix_2014.xlsx', sheet_name='Covariance Matrix')


