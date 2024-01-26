# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize

export_df = pd.read_excel(r'C:\Users\Alex\Desktop\Spyder Py\Working_files\AA_Ptf_Mgmt_DataSXXE46_Monthly_PCT.xlsx', header=0, index_col=(0))


#Step 1 : User input the number of random stocks to be picked

nb_stocks_basket = int(input("Please chose the number of stocks you want in your basket: "))

#Step 2 : create a new df with the randomly selected stocks
basket_df = export_df.sample(n=nb_stocks_basket, axis='columns') 
print(basket_df.head())


#Step 3: define initial weights and backtesting years
backt_years = int(input("number of years to be backtested: "))
nb_months_start = backt_years * 12

rf_rate = float(input("What risk free rate do you want ? ")) # later, need to implement automatically the rfrate
    
initial_weights = 1 / nb_stocks_basket   
weight_df = pd.DataFrame(columns = basket_df.columns)
weight_df.loc[0] = initial_weights
#print(weight_df)

#Step 4 : create a n-Cov matrix
n_cov_matrix = basket_df.cov()
#print(n_cov_matrix)

#Step 5 : average return over the defined period
avg_returns = basket_df.head(nb_months_start).mean()
transp_returns_df = pd.DataFrame(avg_returns).T
#print(transp_returns_df)

#Step 6: creating the ptf metrics 
mon_ptf_ret = (weight_df * transp_returns_df).sum(axis=1).values[0]
ann_ptf_ret = mon_ptf_ret * 12
transp_weight_df = pd.DataFrame(weight_df).T

mon_ptf_var = weight_df @ (n_cov_matrix @ transp_weight_df)
mon_ptf_vol = np.sqrt(mon_ptf_var)
ann_ptf_vol = mon_ptf_vol * np.sqrt(12)
sharpe_ratio = (ann_ptf_ret - rf_rate)/ann_ptf_vol


#---------------------- CHAT GPT advice to perform a sharpe ratio maximiser --- I put it here because no more time 
original_array = np.ones((1, 6))
result_array = original_array * 0.2
# Create a DataFrame df1 with random numbers between -0.5 and 0.5
np.random.seed(42)  # Setting seed for reproducibility
df1 = pd.DataFrame({
    'A': np.random.uniform(-0.5, 0.5, 3),
    'B': np.random.uniform(-0.5, 0.5, 3),
    'C': np.random.uniform(-0.5, 0.5, 3),
    'D': np.random.uniform(-0.5, 0.5, 3),
    'E': np.random.uniform(-0.5, 0.5, 3)
})

# Calculate the covariance matrix
covariance_matrix = df1.cov()

# Define a set of initial weights (replace with your actual weights)
weights = np.array([0.2, 0.3, 0.1, 0.2, 0.2])

# Define the risk-free rate
rf_rate = 0.03

# Define the annual return for the portfolio based on weights
portfolio_return = weights @ df1.mean()

# Define the portfolio variance based on weights and covariance matrix
portfolio_variance = weights @ covariance_matrix @ weights

# Define the Sharpe ratio objective function
def sharpe_ratio(weights, rf_rate, returns, covariance_matrix):
    portfolio_return = weights @ returns
    portfolio_volatility = np.sqrt(weights @ covariance_matrix @ weights)
    sharpe_ratio = (portfolio_return - rf_rate) / portfolio_volatility
    return sharpe_ratio

# Define optimization constraints (weights sum to 1)
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

# Define optimization bounds (weights between 0 and 1)
bounds = tuple((0, 1) for _ in range(len(weights)))

# Run the optimization to maximize the positive Sharpe ratio
result = minimize(lambda w: -sharpe_ratio(w, rf_rate, df1.mean(), covariance_matrix),
                  weights, method='SLSQP', bounds=bounds, constraints=constraints)

# Extract the optimized weights
optimized_weights = result.x

# Calculate the optimized portfolio return and standard deviation
optimized_portfolio_return = optimized_weights @ df1.mean()
optimized_portfolio_std_dev = np.sqrt(optimized_weights @ covariance_matrix @ optimized_weights)

# Calculate the optimized Sharpe ratio
optimized_sharpe_ratio = sharpe_ratio(optimized_weights, rf_rate, df1.mean(), covariance_matrix)