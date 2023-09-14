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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianN
from sklearn import metrics
from scipy import stats
import pandas as pd
from math import sqrt, log, exp, pi
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
from sklearn.feature_selection import RFE
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from numpy import std
from sklearn.tree import plot_tree
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
#import graphviz
from sklearn.preprocessing import scale 
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from xgboost.sklearn import XGBRegressor 
import xgboost.sklearn as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LassoCV
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
import scipy.stats 
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
import pandas as pd

############################################## getting the  ###############################

df1 = pd.read_csv(r'daily\rv_daily.csv') #we delete co1,co3
df2 = pd.read_csv(r'daily\brent_daily.csv',encoding='utf-8', sep=',') #we delete co1,co3
df3 = pd.read_csv(r'daily\ATM.csv',encoding='utf-8', sep=',') #we delete co1,co3
df4 = pd.read_csv(r'daily\CO_daily.csv',encoding='utf-8', sep=',') #we delete co1,co3
# Step 2: Remove leading/trailing whitespace from column names
df1.dropna(inplace=True)
df2.dropna(inplace=True)
df3.dropna(inplace=True)
df4.dropna(inplace=True)

df1.set_index('DATE',inplace=True)
df2.set_index('DATE',inplace=True)
df3.set_index('DATE',inplace=True)
df4.set_index('DATE',inplace=True)
# Merge the DataFrames based on the exact same dates in the DATE column
# Merge the DataFrames based on the intersection of dates
spot_rv_day = pd.merge(df1, df2,left_index=True, right_index=True, how='inner')


iv_spot_rv = pd.merge(spot_rv_day, df3, on='DATE', how='inner')
spot_rv_co = pd.merge(spot_rv_day, df4,left_index=True, right_index=True, how='inner')
iv_spot_rv['RP30'] = iv_spot_rv['ATM_30'] - iv_spot_rv['RV30']
iv_spot_rv['RP60'] = iv_spot_rv['ATM_60'] - iv_spot_rv['RV60']
iv_spot_rv['RP90'] = iv_spot_rv['ATM_90'] - iv_spot_rv['RV90']
iv_spot_rv['RP180'] = iv_spot_rv['ATM_180'] - iv_spot_rv['RV180']
iv_spot_rv['RP270'] = iv_spot_rv['ATM_270'] - iv_spot_rv['RV270']
iv_spot_rv['RP360'] = iv_spot_rv['ATM_360'] - iv_spot_rv['RV360']
iv_spot_rv.index = pd.to_datetime(iv_spot_rv.index)
spot_rv_co.index = pd.to_datetime(spot_rv_co.index)
# Resample to weekly frequency, taking the mean of each week
iv_spot_rv.dropna(inplace=True)
iv_spot_rv_w = iv_spot_rv.resample('W').mean()
iv_spot_rv_m = iv_spot_rv.resample('M').mean()
iv_spot_rv_m.dropna(inplace=True)
iv_spot_rv_w.dropna(inplace=True)
spot_rv_co.dropna(inplace=True)
spot_rv_co_w=spot_rv_co.resample('W').mean()
spot_rv_co_m=spot_rv_co.resample('M').mean()


def calculate_statistics(variable):
    mean = np.mean(variable)
    std_dev = np.std(variable, ddof=0)
    kurt = kurtosis(variable)
    percentile_25 = np.percentile(variable, 25)
    percentile_75 = np.percentile(variable, 75)
    skewness=skew(variable)
    #count=count(variable)
    return [('Mean', mean), ('St  Dev', std_dev), ('Skewness', skewness),
            ('Kurt', kurt), ('25th Percentile', percentile_25), ('75th Percentile', percentile_75)]

variables = {
    'RV30':  iv_spot_rv['RV30'],
    'RV60':  iv_spot_rv['RV60'],
    'RV90':  iv_spot_rv['RV90'],
    'RV180':  iv_spot_rv['RV180'],
    'RV270':  iv_spot_rv['RV270'],
    'RV360':  iv_spot_rv['RV360'],
    'ATM30': iv_spot_rv['ATM_30'],
    'ATM60': iv_spot_rv['ATM_60'],
    'ATM90': iv_spot_rv['ATM_90'],
    'RP30': iv_spot_rv['RP30'],
    'RP60': iv_spot_rv['RP60'],
    'RP90': iv_spot_rv['RP90'],
    'RP180': iv_spot_rv['RP180'],
    'RP270': iv_spot_rv['RP270'],
    'RP360': iv_spot_rv['RP360'],
    'BRENT':  spot_rv_day['BRENT']
}


#print(np.mean(iv_spot_rv_m['RV30']))
variables = {
    'RV30': iv_spot_rv_w['RV30'],
    'RV60': iv_spot_rv_w['RV60'],
    'RV90':  iv_spot_rv_w['RV90'],
    'RV180': iv_spot_rv_w['RV180'],
    'RV270': iv_spot_rv_w['RV270'],
    'RV360': iv_spot_rv_w['RV360'],
    'ATM30': iv_spot_rv_w['ATM_30'],
    'ATM60': iv_spot_rv_w['ATM_60'],
    'ATM90': iv_spot_rv_w['ATM_90'],
    'ATM180': iv_spot_rv_w['ATM_180'],
    'ATM270': iv_spot_rv_w['ATM_270'],
    'ATM360': iv_spot_rv_w['ATM_360'],
    'RP30': iv_spot_rv_w['RP30'],
    'RP60': iv_spot_rv_w['RP60'],
    'RP90': iv_spot_rv_w['RP90'],
    'RP180': iv_spot_rv_w['RP180'],
    'RP270': iv_spot_rv_w['RP270'],
    'RP360': iv_spot_rv_w['RP360'],
    'BRENT':  iv_spot_rv_w['BRENT']
}


variables = {
    'RV30': iv_spot_rv_m['RV30'],
    'RV60': iv_spot_rv_m['RV60'],
    'RV90':  iv_spot_rv_m['RV90'],
    'RV180': iv_spot_rv_m['RV180'],
    'RV270': iv_spot_rv_m['RV270'],
    'RV360': iv_spot_rv_m['RV360'],
    'ATM30': iv_spot_rv_m['ATM_30'],
    'ATM60': iv_spot_rv_m['ATM_60'],
    'ATM90': iv_spot_rv_m['ATM_90'],
    'ATM180': iv_spot_rv_m['ATM_180'],
    'ATM270': iv_spot_rv_m['ATM_270'],
    'ATM360': iv_spot_rv_m['ATM_360'],
    'RP30': iv_spot_rv_m['RP30'],
    'RP60': iv_spot_rv_m['RP60'],
    'RP90': iv_spot_rv_m['RP90'],
    'RP180': iv_spot_rv_m['RP180'],
    'RP270': iv_spot_rv_m['RP270'],
    'RP360': iv_spot_rv_m['RP360'],
    'BRENT':  iv_spot_rv_m['BRENT']
}


# Create a DataFrame to store the descriptive statistics

df = pd.DataFrame(
    columns=['Variable', 'Mean', 'Standard Deviation', 'Skewness', 'Kurtosis', '25th Percentile', '75th Percentile'])
# Iterate over each variable and calculate descriptive statistics
for variable_name, variable_data in variables.items():
    # Calculate descriptive statistics for the current variable
    statistics = {
        'Variable': variable_name,
        'Mean': np.mean(variable_data),
        'Standard Deviation': np.std(variable_data, ddof=0),
        'Skewness': pd.Series(variable_data).skew(),
        'Kurtosis': pd.Series(variable_data).kurtosis(),
        '25th Percentile': np.percentile(variable_data, 25),
        '75th Percentile': np.percentile(variable_data, 75)
    }

    # Append the statistics to the DataFrame
    df = df.append(statistics, ignore_index=True)

# Display the descriptive statistics
print(df)
df.to_excel(r'tablo\descriptive_stats_month.xlsx', index=False)
iv_spot_rv_w.to_excel(r'tablo\'iv_stats_w.xlsx', index=False)
iv_spot_rv_m.to_excel(r'tablo\'iv_stats_m.xlsx', index=False)
iv_spot_rv.to_excel(r'tablo\'iv_stats_d.xlsx', index=False)

spot_rv_co.to_excel(r'tablo\'spot_rv_co.xlsx', index=False)
spot_rv_co_m.to_excel(r'tablo\'spot_rv_co_m.xlsx', index=False)
spot_rv_co_w.to_excel(r'tablo\'spot_rv_co_w.xlsx', index=False)




fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 7))
rv30 = iv_spot_rv_m['RV30']

# Plot histogram for RV30 in the first subplot
axes[0].hist(rv30, bins=30, alpha=0.5, color='blue')
axes[0].set_title('RV30')
axes[0].set_xlabel('Values')
axes[0].set_ylabel('Frequency')

# Plot histogram for RV60 in the second subplot
axes[1].hist(iv_spot_rv_m['RV60'], bins=30, alpha=0.5, color='green')
axes[1].set_title('RV60')
axes[1].set_xlabel('Values')
axes[1].set_ylabel('Frequency')

# Plot histogram for RV60 in the second subplot
axes[2].hist(iv_spot_rv_m['RV180'], bins=30, alpha=0.5, color='red')
axes[2].set_title('RV180')
axes[2].set_xlabel('Values')
axes[2].set_ylabel('Frequency')

# Plot histogram for RV60 in the second subplot
axes[3].hist(iv_spot_rv_m['RV270'], bins=30, alpha=0.5, color='red')
axes[3].set_title('RV270')
axes[3].set_xlabel('Values')
axes[3].set_ylabel('Frequency')

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 7))


# Plot histogram for RV60 in the second subplot
axes[0].hist(iv_spot_rv_m['RP30'], bins=30, alpha=0.5, color='red')
axes[0].set_title('RP30')
axes[0].set_xlabel('Values')
axes[0].set_ylabel('Frequency')

# Plot histogram for RV60 in the second subplot
axes[1].hist(iv_spot_rv_m['RP60'], bins=30, alpha=0.5, color='blue')
axes[1].set_title('RP60')
axes[1].set_xlabel('Values')
axes[1].set_ylabel('Frequency')
# Adjust spacing between subplots

# Plot histogram for RV60 in the second subplot
axes[2].hist(iv_spot_rv_m['RP90'], bins=30, alpha=0.5, color='green')
axes[2].set_title('RP90')
axes[2].set_xlabel('Values')
axes[2].set_ylabel('Frequency')
plt.tight_layout()

# Plot histogram for RV60 in the second subplot
axes[3].hist(iv_spot_rv['RP180'], bins=30, alpha=0.5, color='green')
axes[3].set_title('RP180')
axes[3].set_xlabel('Values')
axes[3].set_ylabel('Frequency')
plt.tight_layout()

# Save the figure as a JPEG file
plt.savefig('histograms.jpg', format='jpeg')


axes[1].plot(iv_spot_rv_m['RP60'], label='RP60')
axes[1].plot(iv_spot_rv_m['RP90'], label='RP90')
axes[1].set_title('RP60 and RP90')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Values')
axes[1].legend()

# Adjust the layout and spacing
plt.tight_layout()



fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))


axes[0].plot(iv_spot_rv_m['RV30'], label='RV30',color='blue', linestyle='--')
axes[0].plot(iv_spot_rv_m['ATM_30'],label='IV30',color='red')
axes[0].set_title('RV30 and IV30')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Values')
#axes[0].axhline(iv_spot_rv_m['RV30'], color='blue', linestyle='--', label='RV30 max')
#axes[0].axhline(iv_spot_rv_m['ATM_30'], color='green', linestyle='--', label='IV30 max')
axes[0].legend()

axes[1].plot(iv_spot_rv_m['RV60'], label='RV60',color='blue', linestyle='--')
axes[1].plot(iv_spot_rv_m['ATM_60'],label='IV60',color='red')
axes[1].set_title('RV60 and IV60')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Values')
axes[1].legend()


axes[2].plot(iv_spot_rv_m['RV90'], label='RV90',color='blue', linestyle='--')
axes[2].plot(iv_spot_rv_m['ATM_90'],label='IV90',color='red')
axes[2].set_title('RV90 and IV90')
axes[2].set_xlabel('Time')
axes[2].set_ylabel('Values')
axes[2].legend()
# Adjust the layout and spacing

plt.tight_layout()

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 6))


axes[0].plot(iv_spot_rv_m['RV180'], label='RV180',color='blue', linestyle='--')
axes[0].plot(iv_spot_rv_m['ATM_180'],label='IV30',color='red')
axes[0].set_title('RV180 and IV80')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Values')
#axes[0].axhline(iv_spot_rv_m['RV30'], color='blue', linestyle='--', label='RV30 max')
#axes[0].axhline(iv_spot_rv_m['ATM_30'], color='green', linestyle='--', label='IV30 max')
axes[0].legend()

axes[1].plot(iv_spot_rv_m['RV270'], label='RV60',color='blue', linestyle='--')
axes[1].plot(iv_spot_rv_m['ATM_270'],label='IV60',color='red')
axes[1].set_title('RV270 and IV270')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Values')
axes[1].legend()


axes[2].plot(iv_spot_rv_m['RV360'], label='RV90',color='blue', linestyle='--')
axes[2].plot(iv_spot_rv_m['ATM_360'],label='IV90',color='red')
axes[2].set_title('RV360 and IV360')
axes[2].set_xlabel('Time')
axes[2].set_ylabel('Values')
axes[2].legend()
# Adjust the layout and spacing
axes[3].plot(iv_spot_rv_m['RP30'], label='RV90',color='blue', linestyle='--')
axes[3].plot(iv_spot_rv_m['RP60'],label='IV90',color='red')
axes[3].set_title('RP60 and RP60')
axes[3].set_xlabel('Time')
axes[3].set_ylabel('Values')
axes[3].legend()
plt.tight_layout()

# Adjust the layout and spacing
plt.tight_layout()

# Display the plot
plt.show()








# Plot time series for RP60 in the second subplot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))

print(spot_rv_co_m)
axes[0].plot(spot_rv_co_m['BRENT'], label='BRENT',color='blue', linestyle='--')
axes[0].plot(spot_rv_co_m['CO1'],label='CO1',color='red')
axes[0].plot(spot_rv_co_m['CO3'],label='CO3',color='green',linestyle='--' )
axes[0].set_title('BRENT,CO1,CO3')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Values')
#axes[0].axhline(iv_spot_rv_m['RV30'], color='blue', linestyle='--', label='RV30 max')
#axes[0].axhline(iv_spot_rv_m['ATM_30'], color='green', linestyle='--', label='IV30 max')
axes[0].legend()

axes[1].plot(spot_rv_co_m['CO6'], label='BRENT',color='blue', linestyle='--')
axes[1].plot(spot_rv_co_m['CO9'],label='CO1',color='red')
axes[1].plot(spot_rv_co_m['CO12'],label='CO3',color='green',linestyle='--' )
axes[1].set_title('CO6,C12,CO12')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Values')
axes[1].legend()

plt.tight_layout()
# Display the plot
plt.show()