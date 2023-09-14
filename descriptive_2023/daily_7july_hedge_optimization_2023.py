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
df2 = pd.read_csv(r'daily\brent_daily.csv', parse_dates=['DATE'])
df3 = pd.read_csv(r'daily\ATM.csv',encoding='utf-8', sep=',') #we delete co1,co3
df4 = pd.read_csv(r'daily\CO_daily.csv',encoding='utf-8', sep=',') #we delete co1,co3

df1.dropna(inplace=True)
df2.dropna(inplace=True)
df3.dropna(inplace=True)
df4.dropna(inplace=True)


df1.set_index('DATE',inplace=True)
df2.set_index('DATE',inplace=True)
df3.set_index('DATE',inplace=True)
df4.set_index('DATE',inplace=True)

print(df2)


# Merge the DataFrames based on the date column
combined_df = pd.merge(df2, df4, left_index=True, right_index=True, how='inner')


df2.index = pd.to_datetime(df2.index)
df4.index = pd.to_datetime(df4.index)

df2 = df2.reindex(df4.index)
# Print the combined DataFrame
df2 = pd.merge(df2, df4, left_index=True, right_index=True, how='inner')
df2['brent_ret'] = 100*(np.log(df2['BRENT']) - np.log(df2['BRENT'].shift(1)))
df2['CO6'] = 100*(np.log(df2['CO6']) - np.log(df2['CO6'].shift(1)))


df2.dropna(inplace=True)


df2m=df2.resample('M').mean()

df2w=df2.resample('W').mean()
df2m.dropna(inplace=True)
df2w=df2w.dropna(inplace=True)





# Merge the DataFrames based on the date column
combined_df = pd.merge(df2, df4, left_on='DATE', right_index=True, how='inner')

# Get the return series from the combined DataFrame
retd = df2['brent_ret']
retco6 = df2['CO6']

# Define the objective function
def objective(h):
    portfolio_returns = retd - h * retco6
    return np.sum(portfolio_returns**2)

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

plt.figure(figsize=(10, 6))
plt.plot(h_values, objective_values)
plt.scatter(optimal_hedge_ratio, objective(optimal_hedge_ratio), color='red', marker='o', label='Optimum')
plt.xlabel('Hedge Ratio (h)')
plt.ylabel('Objective')
plt.title('Objective Function')
plt.legend()
plt.show()

print("Optimal Hedge Ratio:", optimal_hedge_ratio)