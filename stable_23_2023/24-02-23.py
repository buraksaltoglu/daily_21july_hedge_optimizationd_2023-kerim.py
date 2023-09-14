from typing import Any

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
data_finance= pd.read_csv(r"data\finance_6_2_2023.csv", encoding='utf-8', sep=',') #we delet
data_macro =  pd.read_csv(r"data\data_macro_only_13_1_2023.csv", encoding='utf-8', sep=';')
sentiments=   pd.read_csv(r"data/old_sentiments.csv", encoding='utf-8', sep=',')
data_mixed=  pd.read_csv(r"data\data_mixed.csv", encoding='utf-8', sep=',')
##############################################################################################################
#################################### create the data index ###################################################
data_finance.set_index("date",inplace=True)
data_finance_rp.set_index('date',inplace=True)
data_macro.set_index('date',inplace=True)
sentiments.set_index('date',inplace=True)
data_mixed.set_index('date',inplace=True)


############################################## drop covid data ###############################

data_finance.drop(['1.03.2020','1.05.2020','1.04.2020','1.06.2020'],inplace=True)
data_macro.drop(['1.03.2020','1.05.2020','1.04.2020','1.06.2020'],inplace=True)
sentiments.drop(['1.03.2020','1.05.2020','1.04.2020','1.06.2020'],inplace=True)
data_mixed.drop(['1.03.2020','1.05.2020','1.04.2020','1.06.2020'],inplace=True)
#data_finance_rp.drop(['1.03.2020','1.05.2020','1.04.2020','1.06.2020'],inplace=True)
#######################################################################################

X=sentiments.shift(1).dropna()

print(len(X))
y=sentiments['RV30']
y=y.drop(y.index[0])
print(len(y))


X_n=X.drop(['RV30'],axis=1)

model_ols1 = sm.OLS(y, X)
res_ols1 = model_ols1.fit()
res1=res_ols1.summary()
print(res1)

X.to_excel('X_sentiments.xlsx', index = False)
y.to_excel('tablo\y_sentimentshft.xlsx', index = False)

#data_finance.drop(['logret, CO1'],axis=1,inplace=True)
#data_finance_rp=data_finance.drop(['logret','CO1','CO3','CO6','brent'], axis=1,inplace=True)

#######################  Sentiments Regression #########################


def get_regresors():
    regresors = dict ( )
    regresors[ 'cart' ] = DecisionTreeRegressor (max_depth=5)
    regresors[ 'ols' ] = LinearRegression ( )
    regresors[ 'rforest' ] = RandomForestRegressor (max_depth=2, random_state=1)
    regresors[ "gboost" ] = GradientBoostingRegressor ( )
    regresors['XGBoost']= XGBRegressor()
    regresors[ 'LGBoost' ] = LGBMRegressor ( )
    regresors[ 'ridge' ] = Ridge(alpha=1.0)
    regresors['Lasso'] = Lasso(alpha=0.0005)
    return regresors

def evaluate_regresor(regresor, X_n, y):
    selector= RFE(regresor, n_features_to_select=3, step=1)
    selector = selector.fit(X, y)
    return selector

regresors=get_regresors()

####################### Sentiment only RV30 ##############################################################
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = .80 , shuffle=False)

results, names = list ( ), list ( )
b_plot=pd.DataFrame()
scores_sentiments=pd.DataFrame(columns=['regressor','r2','mse','RV'])
for name, regresor in regresors.items ( ):
    model=regresor
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    r2=cross_val_score(model,X, y , cv=5, scoring='r2')
    mse = cross_val_score(model, X, y , cv=5, scoring='neg_mean_squared_error')
    scores_sentiments=scores_sentiments.append({'regressor':name,'r2':r2.mean(),'mse':-mse.mean(),'RV':'RV30'},ignore_index=True)
    b_plot=b_plot.append({'regressor':name,'r2':r2,'mse':mse,'RV':'RV30'},ignore_index=True)
plt.boxplot(-b_plot.r2,labels=regresors.keys(), showmeans=True, notch=True, patch_artist=True)
#plt.savefig('graph\box_rv30_sent.png')
plt.show()

print(scores_sentiments)
b_plot=pd.DataFrame()
plt.show()
plt.title("MSE: RV30 with Only Sentiment Data")
####################### Sentiment only RV60 ##############################################################
y=data_macro.RV60
y=y.drop(y.index[0])
####################### Sentiment only RV60 ##############################################################

X_train, X_test, y_train, y_test = train_test_split(X, y ,train_size = .80 , shuffle=False)
results, names = list ( ), list ( )
for name, regresor in regresors.items ( ):
    model=regresor
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    r2=cross_val_score(model,X, y , cv=5, scoring='r2')
    mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    scores_sentiments=scores_sentiments.append({'regressor':name,'r2':r2.mean(),'mse':-mse.mean(),'RV':'RV60'},ignore_index=True)
    b_plot=b_plot.append({'regressor':name,'r2':r2,'mse':mse,'RV':'RV30'},ignore_index=True)    
plt.boxplot(-b_plot.mse,labels=regresors.keys(), showmeans=True, notch=True, patch_artist=True)
#plt.savefig('graph\rv60_sent.png')
plt.title("MSE: RV60 with Only Sentiment Data")
plt.show()

####################### Sentiment only RV90 ##############################################################

y=data_macro.RV90
y=y.drop(y.index[0])


X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = .80 , shuffle=False)
results, names = list ( ), list ( )
b_plot=pd.DataFrame()
for name, regresor in regresors.items ( ):
    model=regresor
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    r2=cross_val_score(model,X, y, cv=5, scoring='r2')
    mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    scores_sentiments=scores_sentiments.append({'regressor':name,'r2':r2.mean(),'mse':-mse.mean(),'RV':'RV90'},ignore_index=True)
    b_plot=b_plot.append({'regressor':name,'r2':r2,'mse':mse,'RV':'RV30'},ignore_index=True)    
plt.boxplot(-b_plot.mse,labels=regresors.keys(), showmeans=True, notch=True, patch_artist=True)
plt.title("MSE: RV90 with Only Sentiment Data")
#plt.savefig('graph\rv90_sent.png')
plt.show()
scores_sentiments.to_csv('scores\scores_sentiments.csv')


#######################  Finance Only Regression #########################
#X=X_finance.drop(['logret','CO1','CO3','CO6','brent','RP30','RP60','RP90'], axis=1,inplace=True)
X_finance=data_finance.shift(1)
X=X_finance.dropna()
y=data_finance.RV90
y=y.drop(y.index[0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = .80 , shuffle=False)

scores_finance=pd.DataFrame(columns=['regressor','r2','mse','RV'])
for name, regresor in regresors.items ( ):
    model=regresor
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    r2=cross_val_score(model,X, y, cv=5, scoring='r2')
    mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    scores_finance=scores_finance.append({'regressor':name,'r2':r2.mean(),'mse':-mse.mean(),'RV':'RV30'},ignore_index=True)
    b_plot=b_plot.append({'regressor':name,'r2':r2,'mse':mse,'RV':'RV30'},ignore_index=True)    
#plt.boxplot(-b_plot.mse,labels=regresors.keys(), showmeans=True, notch=True, patch_artist=True)
#plt.savefig('graph\box_rv30_fin.png')
#plt.show()

y=data_finance['RV60']
y=y.drop(y.index[0])
X=data_finance.shift(1).dropna()

print("rv60")
print(len(X))
print(len(y))

print('rv60')
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = .80 , shuffle=False)
results, names = list ( ), list ( )
b_plot=pd.DataFrame()
for name, regresor in regresors.items ( ):
    model=regresor
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    r2=cross_val_score(model,X, y, cv=5, scoring='r2')
    mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    scores_finance=scores_finance.append({'regressor':name,'r2':r2.mean(),'mse':-mse.mean(),'RV':'RV60'},ignore_index=True)   
    b_plot=b_plot.append({'regressor':name,'r2':r2,'mse':mse,'RV':'RV30'},ignore_index=True)    
plt.boxplot(-b_plot.mse,labels=regresors.keys(), showmeans=True, notch=True, patch_artist=True)
#plt.savefig('graph\box_rv60_fin.png')
plt.show()
y=data_finance.RV90
X=data_finance.shift(1)
X=X.dropna()
y=y.drop(y.index[0])
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = .80 , shuffle=False)
results, names = list ( ), list ( )
b_plot=pd.DataFrame()
for name, regresor in regresors.items ( ):
    model=regresor
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    r2=cross_val_score(model,X, y, cv=5, scoring='r2')
    mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    scores_finance=scores_finance.append({'regressor':name,'r2':r2.mean(),'mse':-mse.mean(),'RV':'RV30'},ignore_index=True)    
    b_plot=b_plot.append({'regressor':name,'r2':r2,'mse':mse,'RV':'RV90'},ignore_index=True)   
b_plot=b_plot.append({'regressor':name,'r2':r2,'mse':mse,'RV':'RV30'},ignore_index=True)    
#plt.boxplot(-b_plot.mse,labels=regresors.keys(), showmeans=True, notch=True, patch_artist=True)
#plt.savefig('graph\box_rv90_fin.png')
#plt.show()
scores_finance.to_csv('scores_finance.csv')

#######################  Macro Only Regression RV30 #########################
y=data_macro.RV30
y=y.drop(y.index[0])
X=data_macro.shift(1).dropna()



b_plot=pd.DataFrame()
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = .80 , shuffle=False)
results, names = list ( ), list ( )
scores_macro=pd.DataFrame(columns=['regressor','r2','mse','RV'])
for name, regresor in regresors.items ( ):
    model=regresor
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    r2=cross_val_score(model,X, y, cv=5, scoring='r2')
    mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    scores_macro=scores_macro.append({'regressor':name,'r2':r2.mean(),'mse':mse.mean(),'RV':'RV30'},ignore_index=True)
    b_plot=b_plot.append({'regressor':name,'r2':r2,'mse':mse,'RV':'RV30'},ignore_index=True)    
plt.boxplot(-b_plot.mse,labels=regresors.keys(), showmeans=True, notch=True, patch_artist=True)
#plt.savefig('graph\box_rv30_mac.png')
#plt.show()

#######################  Macro Only Regression RV60 #########################
y=data_macro.RV60
y=y.drop(y.index[0])
X=data_macro.shift(1).dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = .80 , shuffle=False)
results, names = list ( ), list ( )
b_plot=pd.DataFrame()
for name, regresor in regresors.items ( ):
    model=regresor
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    r2=cross_val_score(model,X, y, cv=5, scoring='r2')
    mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    scores_macro=scores_macro.append({'regressor':name,'r2':r2.mean(),'mse':mse.mean(),'RV':'RV60'},ignore_index=True)
    b_plot=b_plot.append({'regressor':name,'r2':r2,'mse':mse,'RV':'RV30'},ignore_index=True)    
plt.boxplot(-b_plot.mse,labels=regresors.keys(), showmeans=True, notch=True, patch_artist=True)
#plt.savefig('graph\box_rv60_mac.png')
#plt.show()

#######################  Macro Only Regression RV90 #########################
y=data_macro.RV90
y=y.drop(y.index[0])
X=data_macro.shift(1).dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = .80 , shuffle=False)
results, names = list ( ), list ( )
b_plot=pd.DataFrame()
for name, regresor in regresors.items ( ):
    model=regresor
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    r2=cross_val_score(model,X, y, cv=5, scoring='r2')
    mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    #scores_macro=scores_macro.append({'regressor':name,'r2':scores_macro_r2,'mse':scores_macro_mse,'RV':'RV90'},ignore_index=True)
    b_plot=b_plot.append({'regressor':name,'r2':r2,'mse':mse,'RV':'RV30'},ignore_index=True)    
plt.boxplot(-b_plot.mse,labels=regresors.keys(), showmeans=True, notch=True, patch_artist=True)
#plt.savefig('graph\box_rv90_mac.png')
#plt.show()
scores_macro.to_csv('scores_macro.csv')

#######################  Macro and Finance Regression: RV30 #########################

y=data_mixed.RV30
y=y.drop(y.index[0])
X_mixed=data_mixed.shift(1).dropna()
X=X_mixed.drop('RV30',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = .80 , shuffle=False)
results, names = list ( ), list ( )
b_plot=pd.DataFrame()
scores_mixed=pd.DataFrame(columns=['regressor','r2','mse','RV'])
for name, regresor in regresors.items ( ):
    model=regresor
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    r2=cross_val_score(model,X, y, cv=5, scoring='r2')
    mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    scores_mixed=scores_mixed.append({'regressor':name,'r2':r2.mean(),'mse':mse.mean(),'RV':'RV30'},ignore_index=True)
    b_plot=b_plot.append({'regressor':name,'r2':r2,'mse':mse,'RV':'RV30'},ignore_index=True)    
plt.boxplot(-b_plot.mse,labels=regresors.keys(), showmeans=True, notch=True, patch_artist=True)
#plt.savefig('graph\box_rv30_mix.png')
#plt.show()



#######################  Macro and Finance Regression: RV60 #########################
y=data_mixed.RV60
y=y.drop(y.index[0])

X_mixed=data_mixed.shift(1).dropna()
X=X_mixed.drop('RV60',axis=1)
b_plot=pd.DataFrame()
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = .80 , shuffle=False)
results, names = list ( ), list ( )

for name, regresor in regresors.items ( ):
    model=regresor
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    r2=metrics.r2_score(y_test,y_pred)

    r2=cross_val_score(model,X, y, cv=5, scoring='r2')
    mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    scores_mixed=scores_mixed.append({'regressor':name,'r2':r2.mean(),'mse':mse.mean(),'RV':'RV90'},ignore_index=True)
    b_plot=b_plot.append({'regressor':name,'r2':r2,'mse':mse,'RV':'RV30'},ignore_index=True)    
plt.boxplot(-b_plot.mse,labels=regresors.keys(), showmeans=True, notch=True, patch_artist=True)
#plt.savefig('graph\box_rv60_mix.png')
#plt.show()


#######################  Macro and Finance Regression: RV90 #########################

y=data_mixed.RV90
y=y.drop(y.index[0])

X_mixed=data_mixed.shift(1).dropna()
X=X_mixed.drop('RV90',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = .80 , shuffle=False)
results, names = list ( ), list ( )
b_plot=pd.DataFrame()
for name, regresor in regresors.items ( ):
    amodel=regresor
    amodel.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    r2=cross_val_score(model,X, y, cv=5, scoring='r2')
    scores_mixed_mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    scores_mixed=scores_mixed.append({'regressor':name,'r2':r2.mean(),'mse':scores_mixed_mse.mean(),'RV':'RV90'},ignore_index=True)
    b_plot=b_plot.append({'regressor':name,'r2':r2,'mse':mse,'RV':'RV30'},ignore_index=True)    
plt.boxplot(-b_plot.mse,labels=regresors.keys(), showmeans=True, notch=True, patch_artist=True)
#plt.savefig('graph\box_rv90_mix.png')
#plt.show()
scores_mixed.to_csv('scores_mixed.csv')

#######################  RP Regression from Finance Data #########################
#for col in data_finance_rp.columns:
   # print(col)

y=data_finance['RV30']
y=y.drop(y.index[0])
X=data_finance.shift(1).dropna()
X=X.dropna()
print((y))
print((X))


#datalag1_rp=data_finance_rp.shift(-1)


X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = .80 , shuffle=False)
results, names = list ( ), list ( )
b_plot=pd.DataFrame()
scores_rp=pd.DataFrame(columns=['regressor','scores_r2','scores_mse','RP'])
for name, regresor in regresors.items ( ):
    model=regresor
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    r2=cross_val_score(model,X, y, cv=5, scoring='r2')
    mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    scores_rp=scores_rp.append({'regressor':name,'scores_r2':r2.mean(),'scores_mse':-mse.mean(),'RP':'RP90'},ignore_index=True)
    b_plot=b_plot.append({'regressor':name,'r2':r2,'mse':mse,'RV':'RV30'},ignore_index=True)    
plt.boxplot(-b_plot.mse,labels=regresors.keys(), showmeans=True, notch=True, patch_artist=True)
#plt.savefig('graph\box_rp30.png')
#plt.show()
print(scores_rp)


y=data_finance.RV60
y=y.drop(y.index[0])
X=data_finance.shift(1).dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = .80 , shuffle=False)
results, names = list ( ), list ( )
b_plot=pd.DataFrame()

for name, regresor in regresors.items ( ):
    model=regresor
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    scores_rp_mse = metrics.mean_squared_error(y_test,y_pred)
    r2=cross_val_score(model,X, y, cv=5, scoring='r2')
    mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    scores_rp=scores_rp.append({'regressor':name,'scores_r2':r2.mean(),'scores_mse':-mse.mean(),'RP':'RP90'},ignore_index=True)
    b_plot=b_plot.append({'regressor':name,'r2':r2,'mse':mse,'RV':'RV30'},ignore_index=True)    
plt.boxplot(-b_plot.mse,labels=regresors.keys(), showmeans=True, notch=True, patch_artist=True)
#plt.savefig('graph\box_rp60.png')
#plt.show()
y=data_finance.RV90
X=data_finance.shift(1).dropna()
y=y.drop(y.index[0])
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = .80 , shuffle=False)
results, names = list ( ), list ( )
b_plot=pd.DataFrame()
for name, regresor in regresors.items ( ):
    model=regresor
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    r2=cross_val_score(model,X, y, cv=5, scoring='r2')
    mse = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    scores_rp=scores_rp.append({'regressor':name,'scores_r2':r2.mean(),'scores_mse':-mse.mean(),'RP':'RP90'},ignore_index=True)
    b_plot=b_plot.append({'regressor':name,'r2':r2,'mse':mse,'RV':'RV30'},ignore_index=True)    
plt.boxplot(-b_plot.mse,labels=regresors.keys(), showmeans=True, notch=True, patch_artist=True)
#plt.savefig('graph\box_rp90.png')
#plt.show()
scores_rp.to_csv('RP_Reg_Scores.csv')
############ Pairwise Regression for Macro RV30 #################
columns_res=['' , 'coef', 'std err', 't', 'P>|t|', '[0.025', '0.975]']
results_df=pd.DataFrame()

y=data_finance.RV30
Xlag=data_finance.shift(1).dropna()
y=y.drop(y.index[0])


print(len(Xlag))
print(len(y))

y = list(y)

features_df=pd.DataFrame()

columns_res=['' , 'coef', 'std err', 't', 'P>|t|', '[0.025', '0.975]']
#for column in Xlag:
for column in Xlag:
       # model = MarkovRegression(endog=y, k_regimes=2, trend='n', exog=X,
#                             switching_variance=True)
        #X[column]
        #sm.OLS(y ~ X[column] + data['intercept']).fit()
        Xlag=sm.add_constant(Xlag)
        model_ols_forw = sm.OLS(y, Xlag[column])

        res_ols_forw = model_ols_forw.fit()
        res=res_ols_forw.summary()
        table=pd.DataFrame(res.tables[1])
        features_df=features_df.append(table.iloc[1])
        pred = res_ols_forw.predict(Xlag[column])


features_df.columns=columns_res
features_df.set_index('')
print(features_df)
features_df.to_csv(r'tablo\features_df.csv')
mse = pd.DataFrame([mse])
r2 = pd.DataFrame([r2])

corr = data_finance.corr()


corr.to_csv(r'tablo\corr.csv')
mse.to_csv(r'tablo\mse.csv')
r2.to_csv(r'tablo\r2.csv')






