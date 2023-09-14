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
