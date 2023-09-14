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
from scipy.stats import skew, kurtosis


from sklearn.tree import DecisionTreeClassifier

# to see the columns
# for col in data_finance_rp.columns:
# print(col)
# to make the time series consistent we take RV(t) and X(t-1): (RV(t-1), other features
# We delete the first row of RV(t) and shift the X by 1.
# I show with an example
# df = pd.DataFrame([[1,1],[2,2],[3,3],[4,4]])
# print(df)
#dfx=pd.DataFrame([1,2,3,4])
# print('shift1')
#print(df.shift(1).dropna())
# dfX=dfx.drop(index=0)
# print('dfX')
# print(dfX)
# now we see that dfX is RV(2),RV(3) RV(4) Df RV(1) RV(2) RV(3)

############################################## getting the  ###############################
data_finance_rp = pd.read_csv(r"data\finance_9_1_2023.csv", encoding='utf-8', sep=';') #we delete co1,co3
data_finance= pd.read_csv(r"data\finance_6_2_2023.csv", encoding='utf-8', sep=',') #we delet
data_macro =  pd.read_csv(r"data\data_macro_only_13_1_2023.csv", encoding='utf-8', sep=';')
sentiments=   pd.read_csv(r"data\sentiments.csv", encoding='utf-8', sep=',')
data_mixed=  pd.read_csv(r"data\data_mixed.csv", encoding='utf-8', sep=',')
data_mixed_rp=pd.read_csv(r"data\data_mixed_rp.csv", encoding='utf-8', sep=',') #we delet
##############################################################################################################
#################################### create the data index ###################################################
data_finance.set_index("date",inplace=True)
data_finance_rp.set_index('date',inplace=True)
data_macro.set_index('date',inplace=True)
sentiments.set_index('date',inplace=True)
data_mixed.set_index('date',inplace=True)
data_mixed_rp.set_index('date',inplace=True)
#df.drop(columns=['B', 'C'])
#data_finance.drop([['brent','RP30']],inplace=True)
from openpyxl import Workbook

############################################## drop covid data ###############################

data_finance.drop(['1.03.2020','1.05.2020','1.04.2020','1.06.2020'],inplace=True)
data_macro.drop(['1.03.2020','1.05.2020','1.04.2020','1.06.2020'],inplace=True)
sentiments.drop(['1.03.2020','1.05.2020','1.04.2020','1.06.2020'],inplace=True)
data_mixed.drop(['1.03.2020','1.05.2020','1.04.2020','1.06.2020'],inplace=True)
data_finance_rp.drop(['1.03.2020','1.05.2020','1.04.2020','1.06.2020'],inplace=True)
data_mixed_rp.drop(['1.03.2020','1.05.2020','1.04.2020','1.06.2020'],inplace=True)
#data_finance_rp.drop(['1.03.2020','1.05.2020','1.04.2020','1.06.2020'],inplace=True)
#######################################################################################

X=sentiments.shift(1).dropna()


y=sentiments['RV30']
y=y.drop(y.index[0])
# Create a pandas DataFrame with the variable 'Y'
df = pd.DataFrame(y, columns=['y'])


import pandas as pd
import numpy as np

def calculate_statistics(variable):
    mean = np.mean(variable)
    std_dev = np.std(variable, ddof=0)
    kurt = kurtosis(variable)
    percentile_25 = np.percentile(variable, 25)
    percentile_75 = np.percentile(variable, 75)
    skewness=skew(variable)

    return [('Mean', mean), ('Standard Deviation', std_dev), ('Skewness', skewness),
            ('Kurtosis', kurt), ('25th Percentile', percentile_25), ('75th Percentile', percentile_75)]

# Calculate statistics for each variable

# Create a pandas DataFrame for each variable


print(data_finance)


variables = {
    'RV30': sentiments['RV30'],
    'RV60': sentiments['RV60'],
    'RV90': sentiments['RV90'],
    'ATM30': data_finance['ATM30'],
    'ATM60': data_finance['ATM60'],
    'ATM90': data_finance['ATM90'],
    'RP30': data_finance['RP30'],
    'RP60': data_finance['RP60'],
    'RP90': data_finance['RP90']
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
df.to_excel('descriptive_stats.xlsx', index=False)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
rv30 = data_finance['RV30']

# Plot histogram for RV30 in the first subplot
axes[0].hist(rv30, bins=30, alpha=0.5, color='blue')
axes[0].set_title('RV30')
axes[0].set_xlabel('Values')
axes[0].set_ylabel('Frequency')

# Plot histogram for RV60 in the second subplot
axes[1].hist(data_finance['RV60'], bins=30, alpha=0.5, color='green')
axes[1].set_title('RV60')
axes[1].set_xlabel('Values')
axes[1].set_ylabel('Frequency')

# Plot histogram for RV60 in the second subplot
axes[2].hist(data_finance['ATM30'], bins=30, alpha=0.5, color='red')
axes[2].set_title('ATM30')
axes[2].set_xlabel('Values')
axes[2].set_ylabel('Frequency')

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))


# Plot histogram for RV60 in the second subplot
axes[0].hist(data_finance['RP30'], bins=30, alpha=0.5, color='red')
axes[0].set_title('RP30')
axes[0].set_xlabel('Values')
axes[0].set_ylabel('Frequency')

# Plot histogram for RV60 in the second subplot
axes[1].hist(data_finance['RP60'], bins=30, alpha=0.5, color='blue')
axes[1].set_title('RP60')
axes[1].set_xlabel('Values')
axes[1].set_ylabel('Frequency')
# Adjust spacing between subplots

# Plot histogram for RV60 in the second subplot
axes[2].hist(data_finance['RP90'], bins=30, alpha=0.5, color='green')
axes[2].set_title('RP90')
axes[2].set_xlabel('Values')
axes[2].set_ylabel('Frequency')
plt.tight_layout()

# Save the figure as a JPEG file
plt.savefig('histograms.jpg', format='jpeg')

# Display the plot
plt.show()