

import statsmodels.formula.api as sm
import statsmodels.api as sm
import matplotlib
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from plotnine import ggplot, aes, geom_line
from plotnine import *
#import yfinance as yf
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from finta import TA
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import xgboost as xgb
from sklearn.metrics import confusion_matrix
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
from sklearn import metrics
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LassoCV
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier

# read data futures and options 2008 2022 reuters brent daily 1987-2022 in 2022 1:4 months crisis deleted

data = pd.read_csv(r'data\CO_IV_08-22_ATM_RR75_25.csv', encoding='utf-8', sep=';', index_col ='date',parse_dates=['date'])
datam = pd.read_csv(r'data\CO_IV_08-22_ATM_RR75_25.csv', encoding='utf-8', sep=';', parse_dates=['date'])
#datamdummy = pd.read_csv(r'data\datamdummy.csv', encoding='utf-8', sep=';', parse_dates=['date'])
#databrent = pd.read_csv(r'data\brent_87_2022.csv', encoding='utf-8', sep=';', parse_dates=['Date'])

####################################date index ##########################
#data.set_index('date', inplace=True)
#databrent.set_index('Date', inplace=True)
datam.index = pd.to_datetime(datam.index)
datam=datam.dropna() # drop if NA
#datam=datam.drop(datam.index[133:3300])

######################transform the data %return linear diff ###############


#datareturn=datam.pct_change()
#datareturn = datareturn.dropna()

#################### calculating log returns ######################

datam['log_retsq']= (np.log(datam['brent'])-np.log(datam['brent']).shift(1))**2
datam['log_rtn'] = np.log(datam['brent']).diff()
#datam['SMA30'] = datam['log_rtn'].rolling(30).mean()
#datam['4dayEWM'] = datam['log_rtn'].ewm(span=4, adjust=False).mean()
####################### implied SKEW  25 delta RR for skew ##########
datam['25RR_30'] = datam['D_30_25']-datam['D_30_75']  # 25 RR 60 days
datam['25RR_60'] = datam['D_60_25']-datam['D_60_75']
datam['25RR_90'] = datam['D_90_25']-datam['D_90_75']
datam['25RR_180'] = datam['D_180_25']-datam['D_180_75']
####################### term structure of crude volatilty       60-30, 90-30 etc ##########
datam['vol60_30'] = datam['ATM60']-datam['ATM30']
datam['vol90_30'] = datam['ATM90']-datam['ATM30']
datam['vol180_30'] = datam['ATM180']-datam['ATM30']



####################### slope of crude forward curve  CO6-CO3 etc    ##########
datam['CO31'] = datam['CO3']-datam['CO1']
datam['CO61'] = datam['CO6']-datam['CO1']
datam['CO63'] = datam['CO6']-datam['CO3']

###################### calculating RVariance and Rvolatility for monthly data ##############
window30 = 21  # trading days in rolling window monthly
window60=42
window90=63

dpy = 252  # trading days per year
ann_factor = dpy / window30

datam['RVAR30'] = datam['log_retsq'].rolling(window30).mean() * dpy
datam['RVAR60'] = datam['log_retsq'].rolling(window60).mean() * dpy
datam['RVAR90'] = datam['log_retsq'].rolling(window90).mean() * dpy

datam['RV30'] = np.sqrt(datam['RVAR30'])*100
datam['RV60'] = np.sqrt(datam['RVAR60'])*100
datam['RV90'] = np.sqrt(datam['RVAR90'])*100
datam['RV30ema']=datam['RV30'].rolling(window30).mean()
print(datam)
datam.dropna(inplace=True)

print('-----datam na')

RV60=datam['RV60']


###################### converting daily to weekly and monthly   ##############

datad = (datam.resample('1D').mean())
dataw= (datam.resample('1W').mean())
datam= (datam.resample('1M').mean())


###################### taking lags intercept   ##############

datalag1=datam.shift(1)
datalag1.dropna()
datalag2=datam.shift(2)
datalag2.dropna()
datam.dropna(inplace=True)
datam['intercept'] = 1


datam['RV30lag1']=datam['RV30'].shift(-1) # Rvolatilities lags1 2
datam['RV30lag2']=datam['RV30'].shift(-2)
datam['RV30lag3']=datam['RV30'].shift(-3)
datam['ATM30lag1']=datam['ATM30'].shift(-1)
datam['RV60lag1']= datam['RV60'].shift(-1) # Rvolatilities lags1
datam['RV90lag1']= datam['RV90'].shift(-1) # Rvolatilities lags1
datam['25RR_30lag1']= datam['25RR_30'].shift(-1) # Rvolatilities lags1
datam['25RR_60lag1']= datam['25RR_60'].shift(-1) # Rvolatilities lags1
datam['25RR_90lag1']= datam['25RR_90'].shift(-1) # Rvolatilities lags1
datam = datam.dropna()
datam.info()
### dropping unwanted columns from data frame
#datam.drop(datam.columns[[0,1,2,3,4,5,6,7,8,9,10,11]], axis=1, inplace=True)
RV30lag1=datam['RV30lag1']
# writting the data into a file




######### dropping crisis and outliers from the data set #######

datam=datam.drop(datam.index[133:138]) # deleting crisis months


######### target y: RV30, feature set X  #######
y=datam['RV30']
datam['riskpremium']=datam['ATM30']-datam['RV30']
datam['riskpremiumema']=datam['riskpremium'].rolling(window30).mean()

datam['target'] = np.where(datam.RV30 > datam['ATM30'], 1, 0)
datam['target'] = np.where(datam.riskpremium > datam['riskpremiumema'], 1, 0)
datam['target'] = np.where(datam.RV30 > datam['ATM30'], 1, 0)
datam['target'] = np.where(datam.riskpremium > datam['riskpremium'].mean(), 1, 0)
print((datam['riskpremium']).std())
y=datam['target']
X=datam[['RV30lag1','RV30lag2','RV60lag1','ATM180','ATM60','ATM30lag1','vol90_30','vol180_30','25RR_30','25RR_60','25RR_90','CO31','CO61']]
#X=datam[['intercept','RV30lag1' , 'RV30lag2', 'ATM30lag1', 'vol90_30', 'CO31','CO61', '25RR_60','25RR_90']]
Xraw=X
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = .80 , shuffle=False)
pca = PCA()

X = pca.fit_transform(scale(X))
model = xgb.XGBClassifier()
model.fit(X_train,y_train)
print(); print(model)



y_pred_xgb = model.predict(X_test)

#log_pca_reg=sm.Logit(y,X_reduced[:,:4]).fit()
log_pca_reg=sm.Logit(y_train,X_train).fit()
y_pred = log_pca_reg.predict(X_test)
y_pred = list(map(round, y_pred))


# accuracy score of the model
print('Test accuracy  = ', accuracy_score(y_test, y_pred))
print('Test accuracy1 = ', accuracy_score(y_test, y_pred_xgb))
# comparing original and predicted values of y
#print('Predictions :', y_pred)
#log_reg = sm.Logit(y, X).fit()
#print(log_pca_reg.summary())
#print(log_reg.summary())
#X = (X - X.mean ( )) / X.std ( ) # normalize for lasso and PCA


#score = log_pca_reg.score(X, y)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix : \n", cm)
print('The accuracy of the Logistic Regression is', metrics.accuracy_score(y_pred,y_test))
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(cmn, annot=True, fmt='.2f')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(y_test,y_pred_xgb))
plt.title(all_sample_title, size = 15);

plt.show()
######### setting the training and test sets  #########################
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = .80 , shuffle=False)
X_unc=datam[['intercept','RV30lag1','RV60lag1','RV90lag1']]
X_forw=datam[['intercept','RV30lag1','RV60lag1','RV90lag1', 'CO31','CO63' ]]

X_forw1=datam[['intercept','RV30lag1','RV60lag1','RV90lag1', 'CO31' ]]
X_forw2=datam[['intercept','RV30lag1','RV60lag1','RV90lag1', 'CO63' ]]

X_skew=datam[[ 'intercept','RV30lag1','RV60lag1','RV90lag1', '25RR_30lag1','25RR_60lag1','25RR_90lag1']]
X_skew1=datam[[ 'intercept','RV30lag1','RV60lag1','RV90lag1', '25RR_30','25RR_60']]
X_skew2=datam[[ 'intercept','RV30lag1','RV30lag2','RV60lag1', '25RR_30']]
#X_skew3=datam[[ 'intercept','RV30lag1','RV30lag2','RRV60lag1', '25RR_60']]

X_IV=datam[[ 'intercept','ATM30lag1']]
X_volterm=datam[[ 'intercept','RV30lag1','RV30lag2','RV60lag1','vol180_30','vol60_30','vol90_30']]
X_volterm1=datam[[ 'intercept','RV30lag1','RV30lag2','RV60lag1','vol60_30']]
X_volterm2=datam[[ 'intercept','RV30lag1','RV30lag2','RV60lag1','vol90_30']]
X_volterm3=datam[[ 'intercept','RV30lag1','RV30lag2','RV60lag1','vol180_30']]

 
seed = 7
models = []
models.append(('LogReg', LogisticRegression(random_state=seed)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNNC', KNeighborsClassifier()))
models.append(('DTreeC', DecisionTreeClassifier()))
#models.append(('GaussianNB', GaussianNB()))
models.append(('RForestC', RandomForestClassifier()))
models.append(('ETreesC',ExtraTreesClassifier(random_state=seed)))
models.append(('AdaBC',AdaBoostClassifier(DecisionTreeClassifier(random_state=seed),random_state=seed,learning_rate=0.1)))
models.append(('SVM',svm.SVC(random_state=seed)))
models.append(('GBoostC',GradientBoostingClassifier(random_state=seed)))
models.append(('MLPC',MLPClassifier(random_state=seed)))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'



for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name),
plt.boxplot(results,labels=names, showmeans=True, notch=True, patch_artist=True)
plt.show()
plt.savefig('graph\cv_results_box.png')
########################
pdfs=list()
mean_std=pd.DataFrame(columns=['model','mean','std'])

pdfs=list()
mean_std=pd.DataFrame(columns=['model','mean','std'])
results=[]

for name, model in models:
    csv_df=pd.DataFrame(columns=['cm','acc'])
    model.fit(X_train,y_train)
    print(model)
    
    y_pred = model.predict(X)
    y_pred = list(map(round, y_pred))
    y_p_train=model.predict(X_train)
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix : \n", cm)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))

    
    
    
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    mean_std=mean_std.append({'model':name,'mean':cv_results.mean(),'std': cv_results.std()},ignore_index=True)
    results.append(cv_results)
    
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = '{0}\nAccuracy Score: {1}\nCV Average: {2}'.format(name,accuracy_score(y,y_pred),cv_results.mean())
    plt.title(all_sample_title, size = 15);
    plt.savefig('graph\figure_%s.pdf' % name,format='pdf')
    plt.show()
    pdfs.append('graph\figure_%s.pdf' % name)
    csv_df=csv_df.append({'cm': cm,'acc_test': accuracy_score(y,y_pred),'acc_train':accuracy_score(y,y_p_train) },ignore_index=True)
    csv_df.to_csv(r'tablo\cm_acc_%s.csv'%name)
mean_std.to_csv(r'tablo\mean_std.csv')
from PyPDF2 import PdfMerger
merger = PdfMerger()
for pdf in pdfs:
    merger.append(pdf)

merger.write("tablo\result.pdf")
merger.close()