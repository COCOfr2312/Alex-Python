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

#Step 3: define initial weights and backtesting years
backt_years = int(input("number of years to be backtested: "))
nb_months_start = backt_years * 12


#Step 4: define the Risk Free Rate
rf_rate = float(input("What risk free rate do you want ? ")) # later, need to implement automatically the rfrate

#Step 5 : stocks weights and n-Cov matrix
weight = np.full(len(basket_df.columns), 1/len(basket_df.columns), dtype=float)
n_cov_matrix = basket_df.cov()

#Step 6: create the empty dataframe to store our mean and sharpe ratio results
result_basket_df = pd.DataFrame(columns=basket_df.columns)
sharpe_ratio_df = pd.DataFrame(columns=['Sharpe Ratio'])

#--------------LOOP STARTS HERE----------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

for i in range(nb_months_start,len(basket_df)):
    subset_df = basket_df.iloc[i-nb_months_start:i]
    mean_return = subset_df.mean()
    
    #Step 5 : average return over the defined period
    #mean_return = basket_df.head(nb_months_start).mean()
    portfolio_return = weight @ mean_return
    portfolio_variance = weight @ n_cov_matrix @ weight
    
    
    def sharpe_ratio_def(weight, rf_rate, avg_returns, n_cov_matrix):
        portfolio_return = weight @ avg_returns * 12
        portfolio_volatility = np.sqrt(weight @ n_cov_matrix @ weight) * np.sqrt(12)
        sharpe_ratio_def = (portfolio_return - rf_rate) / portfolio_volatility
        return sharpe_ratio_def
    
    
    # Define optimization constraints (weights sum to 1)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    
    # Define optimization bounds (weights between 0 and 1)
    bnds = tuple((0, 1) for _ in range(len(weight)))
    
    
    # Run the optimization to maximize the positive Sharpe ratio
    result = minimize(lambda w: -sharpe_ratio_def(w, rf_rate, mean_return, n_cov_matrix),
                      weight, method='SLSQP', bounds=bnds, constraints=constraints)
    
    
    # Extract the optimized weights
    optimized_weights = result.x
    
    # Calculate the optimized portfolio return and standard deviation
    optimized_portfolio_return = optimized_weights @ mean_return *12 # good type
    optimized_portfolio_std_dev = np.sqrt(optimized_weights @ n_cov_matrix @ optimized_weights) *np.sqrt(12)
    
    # Calculate the optimized Sharpe ratio
    optimized_sharpe_ratio = sharpe_ratio_def(optimized_weights, rf_rate, mean_return, n_cov_matrix)
    
    #print(optimized_portfolio_return)
    #print(optimized_portfolio_std_dev)
    #print(optimized_weights)
    
    #ADDING THE OPTIMIZED WEIGHTS TO OUR BLANK THIRD PARTY DF
    #result_basket_df = result_basket_df.append(optimized_weights, ignore_index= False)
    
    #new_basket_df= pd.DataFrame(columns= basket_df.columns)
    row_date = pd.to_datetime(basket_df.index[i])
    result_basket_df.loc[row_date]= optimized_weights
    sharpe_ratio_df.loc[row_date] = optimized_sharpe_ratio
   

final_df = pd.concat([result_basket_df,sharpe_ratio_df], axis=1)
