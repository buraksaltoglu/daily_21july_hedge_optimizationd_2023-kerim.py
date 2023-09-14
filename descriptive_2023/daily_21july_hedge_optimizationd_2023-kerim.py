from typing import Any
#import openpyxl
import statsmodels.formula.api as sm
import statsmodels.api as sm
import matplotlib
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from plotnine import ggplot, aes, geom_line
from plotnine import *
import yfinance as yf
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
#from finta import TA
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from random import uniform
import plotnine
from plotnine import ggplot, aes, geom_line
from plotnine import *
from plotnine.data import economics
from plotnine import ggplot, aes, geom_line,geom_density
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import mean_squared_error
from mpl_toolkits import mplot3d
from datetime import datetime
from itertools import chain
from matplotlib import cm
from sklearn.neural_network import MLPRegressor
from datetime import datetime
from sklearn import model_selection
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import RFE
from scipy.stats import norm
from scipy.stats import norm
import numpy as np
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from scipy.stats import norm
from scipy.optimize import minimize
# ...

from sklearn.linear_model import LassoCV
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
import scipy.stats 
warnings.filterwarnings('ignore')
from scipy.stats import norm
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

############################################## getting the  ###############################

df1 = pd.read_csv(r'daily\rv_daily.csv') #we delete co1,co3
#df2 = pd.read_csv(r'daily\brent_daily.csv',encoding='utf-8', sep=',') #we delete co1,co3
df2 = pd.read_csv(r'daily\brent_daily.csv')
df3 = pd.read_csv(r'daily\ATM.csv',encoding='utf-8',parse_dates=['DATE'], sep=',') #we delete co1,co3
df4 = pd.read_csv(r'daily\CO_daily.csv',encoding='utf-8', parse_dates=['DATE'], sep=',') #we delete co1,co3

df1.dropna(inplace=True)
df2.dropna(inplace=True)
df3.dropna(inplace=True)
df4.dropna(inplace=True)

df1.set_index('DATE',inplace=True)
df2.set_index('DATE',inplace=True)
df3.set_index('DATE',inplace=True)
df4.set_index('DATE',inplace=True)




# Merge the DataFrames based on the date column
combined_df = pd.merge(df2, df4, left_index=True, right_index=True, how='inner')


df2.index = pd.to_datetime(df2.index)
df4.index = pd.to_datetime(df4.index)

df2 = df2.reindex(df4.index)

# Print the combined DataFrame
df2 = pd.merge(df2, df4, left_index=True, right_index=True, how='inner')
df2['brent_ret'] = 100*(np.log(df2['BRENT']) - np.log(df2['BRENT'].shift(1)))
df2['CO6ret'] = 100*(np.log(df2['CO6']) - np.log(df2['CO6'].shift(1)))


df2.dropna(inplace=True)


df2m=df2.resample('M').mean()

df2w=df2.resample('W').mean()
df2m.dropna(inplace=True)
df2w.dropna(inplace=True)



# Merge the DataFrames based on the date column
combined_df = pd.merge(df2, df4, left_on='DATE', right_index=True, how='inner')

# Get the return series from the combined DataFrame
retd = df2['brent_ret']
retco6 = df2['CO6ret']
print('correlation')
print(retco6.corr(retd))
k=10
window_size = 126
S_t_f = df2['brent_ret'].rolling(window=window_size).mean().shift(-window_size)
S_t_f1 = df2['BRENT'].rolling(window=window_size).mean().shift(-window_size)
# Drop NaN values from the resulting series
S_t_f = S_t_f.dropna()
print(df2.head(2))
# Add future_5day_avg_brent_ret to df
df2['S_t_f'] = S_t_f
window_size = 21
df2['S_t_f1'] = df2['BRENT'].rolling(window=window_size).mean().shift(-window_size)

window_sizes = [21, 63]
rolling_averages = [df2['BRENT'].rolling(window=window_size).mean().shift(-window_size) for window_size in window_sizes]

# Plot the rolling averages
plt.figure(figsize=(10, 6))
plt.plot(df2.index, rolling_averages[0], label='Window = 21')
plt.plot(df2.index, rolling_averages[1], label='Window = 63')
plt.axhline(df2['BRENT'].mean(), color='r', linestyle='--', label='Mean of BRENT')
plt.xlabel('Date')
plt.ylabel('Rolling Average')
plt.title('Rolling Averages of BRENT')
plt.legend()
plt.show()


# Print the result

ret_6m_avg = retd.rolling(window=2).mean().shift(-2)

# Drop NaN values from both series to ensure they have the same size
ret_6m_avg = ret_6m_avg.dropna()
retco6= retco6.dropna()

# Ensure both series have the same size by considering only overlapping dates
common_dates = ret_6m_avg.index.intersection(retco6.index)
ret6m = ret_6m_avg.loc[common_dates]
retco6 = retco6.loc[common_dates]







comparison_df = pd.concat([retd, ret6m], axis=1, keys=['retd', 'ret6m'])
print("Comparison of the first 10 rows of retco6 and retd:")


#print(six_month_future_avg_retd)
# Define the objective function
def objective(h):
    portfolio_returns = ret6m - h * retco6
    return np.mean(portfolio_returns**2)

# Define the constraint that h should be between 0 and 1
constraint = ({'type': 'ineq', 'fun': lambda h: h}, {'type': 'ineq', 'fun': lambda h: 1 - h})

# Set the initial guess for h
initial_guess = 0.5

# Perform the optimization
result = minimize(objective, initial_guess, constraints=constraint)

# Get the optimal hedge ratio
optimal_hedge_ratio = result.x[0]

# Plot the objective function
h_values = np.linspace(0, 1, 100)
objective_values = [objective(h) for h in h_values]

plt.figure(figsize=(14, 10))
plt.plot(h_values, objective_values)
plt.scatter(optimal_hedge_ratio, objective(optimal_hedge_ratio), color='red', marker='o', label='Optimum')
plt.xlabel('Hedge Ratio (h)')
plt.ylabel('Optimal Hedge Ratio by  Minimizing Hedge Variance ')
plt.title('Choosing the Optimal Hedge Ratio ')
plt.legend()
#plt.show()

print("Optimal Hedge Ratio:", optimal_hedge_ratio)
# regression is not used at this stage

#df2m['intercept'] = 1
#X = df2m[['CO6ret', 'intercept']]

# Add an intercept term to X
#X['intercept'] = 1

# Create the dependent variable y
#y = df2m['brent_ret']

# Fit the linear regression model
#model = sm.OLS(y, X)
#results = model.fit()
print('correlation')
print(retco6.corr(ret6m))
# Define the initial guess for h
h_initial = 0.25

# Define the range of h values to consider
hs = np.linspace(0, 1, num=100)

# Initialize a list to store VaR values
var_values = []

# Iterate over each h value
for h in hs:
    # Calculate portfolio returns for current h
    portfolio_returns = df2['brent_ret'] - h * df2['CO6ret']
    portfolio_returns = ret6m - h * retco6
    # Sort portfolio returns in ascending order
    sorted_returns = np.sort(portfolio_returns)

    # Calculate VaR as the 5th percentile of sorted returns
    var = np.percentile(sorted_returns, 95)

    # Append VaR to the list
    var_values.append(var)

# Find the optimal hedge ratio that minimizes VaR
optimal_h = hs[np.argmin(var_values)]
print("Optimal Hedge Ratio:", optimal_h)

# Plot VaR against h
plt.figure(figsize=(12, 6))
plt.plot(hs, var_values, color='blue', linewidth=2)
plt.xlabel('h')
plt.ylabel('Value at Risk (VaR)')
plt.title('Optimizing Hedge Ratio via VaR ')
plt.grid(True)
plt.axvline(x=optimal_h, color='red', linestyle='--', label='Optimal Hedge Ratio')
plt.legend()
#plt.show()

# Define the initial guess for h
h_initial = 0.25

# Define the range of h values to consider
hs = np.linspace(0, 1, num=100)

# Initialize a list to store ES values
es_values = []

# Iterate over each h value
for h in hs:
    # Calculate portfolio returns for current h
    portfolio_returns = df2['brent_ret'] - h * df2['CO6ret']
    portfolio_returns = ret6m - h * retco6
    # Sort portfolio returns in ascending order
    sorted_returns = np.sort(portfolio_returns)

    # Calculate VaR as the 5th percentile of sorted returns
    var = np.percentile(sorted_returns, 95)

    # Calculate ES as the mean of sorted returns beyond the VaR level
    es = sorted_returns[sorted_returns >= var].mean()

    # Append ES to the list
    es_values.append(es)

# Find the optimal hedge ratio that minimizes ES
optimal_h = hs[np.argmin(es_values)]
print("Optimal Hedge Ratio:", optimal_h)

print("Optimal Hedge Ratio with ES:", optimal_h)

# Calculate portfolio returns with optimal h
portfolio_returns_optimal = df2['brent_ret'] - optimal_h * df2['CO6ret']


# Plot brent_ret and portfolio_returns with optimal h
plt.figure(figsize=(12, 6))
plt.plot(df2.index, df2['brent_ret'], color='blue', linewidth=2, label='brent_ret')
plt.plot(df2.index, portfolio_returns_optimal, color='red', linewidth=2, label='portfolio_returns (Optimal h)')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.title('brent_ret vs. portfolio_returns')
plt.grid(True)
plt.legend()
#plt.show()
# Define the initial guess for h
h_initial = 0.25

# Define the range of h values to consider
hs = np.linspace(0, 1, num=100)

# Initialize a list to store ES values
es_values = []

# Iterate over each h value
for h in hs:
    # Calculate portfolio returns for current h
    portfolio_returns = df2['brent_ret'] - h * df2['CO6ret']

    # Sort portfolio returns in ascending order
    sorted_returns = np.sort(portfolio_returns)

    # Calculate VaR as the 5th percentile of sorted returns
    var = np.percentile(sorted_returns, 95)

    # Calculate ES as the mean of sorted returns beyond the VaR level
    es = sorted_returns[sorted_returns >= var].mean()

    # Append ES to the list
    es_values.append(es)

# Find the optimal hedge ratio that minimizes ES
optimal_h = hs[np.argmin(es_values)]
print("Optimal Hedge Ratio CVaR:", optimal_h)

# Plot ES against h
plt.figure(figsize=(12, 6))
plt.plot(hs, es_values, color='blue', linewidth=2)
plt.xlabel('h')
plt.ylabel('Expected Shortfall (ES)')
plt.title('Optimal Hedge Ratio with CVaR: h*')
plt.grid(True)
plt.axvline(x=optimal_h, color='red', linestyle='--', label='Optimal Hedge Ratio')
plt.legend()
#plt.show()
# Plot histogram of brent_ret and portfolio_returns_optimal on the same plot
plt.figure(figsize=(12, 6))

plt.hist(df2['brent_ret'], bins=100, color='blue', alpha=0.7, label='brent_ret')
plt.hist(portfolio_returns_optimal, bins=100, color='red', alpha=0.7, label=f'portfolio_returns (h={optimal_h:.2f})')

plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.title('Histogram of brent_ret and portfolio_returns')
#plt.legend()

#plt.show()

# Calculate descriptive statistics
brent_ret_stats = df2['brent_ret'].describe()
portfolio_returns_optimal_stats = portfolio_returns_optimal.describe()

print("Descriptive Statistics - brent_ret:")
print(brent_ret_stats)
print("\nDescriptive Statistics - portfolio_returns (Optimal h):")
print(portfolio_returns_optimal_stats)
print('hedge performance')
print('------------------------------------------------------------')

varunhedged = np.percentile(df2['brent_ret'], 99)
varhedged = np.percentile(portfolio_returns_optimal, 99)


print("St dev of unhedged:", np.std(df2['brent_ret']))
sorted_returns = np.sort(portfolio_returns)

    # Calculate VaR as the 5th percentile of sorted returns

var_hedge = np.percentile(sorted_returns, 95)

    # Calculate ES as the mean of sorted returns beyond the VaR level
es_hedged = sorted_returns[sorted_returns >= var_hedge].mean()

print("hedged es",es_hedged)
print("St dev of hedged:", np.std(portfolio_returns_optimal))

print("VaR of hedged:", varhedged)
print("VaR of unhedged:", varunhedged)

print(varunhedged)
print(varhedged)
print(1-varhedged/varunhedged)

print("Hedge Effectiveness")
print('1-(VaR_hedged/VaR_unhedged)')
print(1-(varhedged/varunhedged))
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize

import numpy as np
from scipy.optimize import minimize

# Set the random seed for reproducibility
np.random.seed(42)

# Define the number of trading days and stocks
num_days = 1000
num_stocks = 4

# Generate random returns for four stocks
returns = np.random.normal(loc=0.001, scale=0.02, size=(num_days, num_stocks))

# Create a pandas DataFrame for the returns
data = pd.DataFrame(returns, columns=['XOM', 'CVX', 'BP', 'RDS-A'])

# Calculate cumulative returns
cumulative_returns = (1 + data).cumprod()

# Define the objective function for expected return
def objective(weights):
    return -np.sum(data.mean() * weights)

# Define the constraint function for VaR
def constraint(weights):
    portfolio_returns = np.sum(data * weights, axis=1)
    return np.mean(portfolio_returns) - np.percentile(portfolio_returns, 5)

# Define the initial guess for weights
initial_weights = np.ones(num_stocks) / num_stocks

# Define the weight bounds
bounds = [(0, 1)] * num_stocks

# Define the optimization problem
problem = {
    'fun': objective,
    'constraints': [
        {'type': 'ineq', 'fun': constraint}
    ],
    'bounds': bounds,
    'x0': initial_weights
}

# Solve the optimization problem
result = minimize(**problem)

# Print the optimized weights
optimal_weights = result.x
for stock, weight in zip(data.columns, optimal_weights):
    print(f"{stock}: {weight:.4f}")
