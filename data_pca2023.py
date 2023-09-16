import os
import copy
import csv
import re
import warnings
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
sns.set_style('white')

warnings.filterwarnings('ignore')


# to make the time series consistent we take RV(t) and X(t-1): (RV(t-1), other features
# We delete the first row of RV(t) and shift the X by 1.
# I show with an example
# df = pd.DataFrame([[1,1],[2,2],[3,3],[4,4]])
# print(df)
# dfx=pd.DataFrame([1,2,3,4])
# print('shift1')
# print(df.shift(1).dropna())
# dfX=dfx.drop(index=0)
# print('dfX')
# print(dfX)
# now we see that dfX is RV(2),RV(3) RV(4) Df RV(1) RV(2) RV(3)

# getting the data ####################################################
# data_finance_rp = pd.read_csv(r"data\finance_9_1_2023.csv", encoding='utf-8', sep=';')  # we delete co1,co3
# data_finance = pd.read_csv(r"data\finance_6_2_2023.csv", encoding='utf-8', sep=',')  # we delete
# #data_macro = pd.read_csv(r"data\data_macro_only_1_6_2023.csv", encoding='utf-8', sep=',')
# data_macro = pd.read_csv(r"data\data_macro_only_13_1_2023.csv", encoding='utf-8', sep=';')
# sentiments = pd.read_csv(r"data/sentiments.csv", encoding='utf-8', sep=',')
# data_mixed = pd.read_csv(r"data\data_mixed.csv", encoding='utf-8', sep=',')

# Specify the directory where the files are located
directory = "new_data"

file_dictionary = {}

# Process each file in the directory
for file in os.listdir(directory):
    if file.endswith(".xlsx"):
        file_name = file.lower().replace('.xlsx','')
        file_path = os.path.join(directory, file)
        data = pd.read_excel(file_path)
        file_dictionary[file_name] = data

for key_name in file_dictionary.keys():
    print(key_name)

# Example
monthly_dataset_sentiment_92_filtered = file_dictionary['monthly_dataset_sentiment_92_filtered']
monthly_dataset_sentiment_92_filtered.set_index('date', inplace=True)

##############################################################################################################

# create the data index ###################################################
# data_finance.set_index("date", inplace=True)
# data_finance_rp.set_index('date', inplace=True)
# data_macro.set_index('date', inplace=True)
# sentiments.set_index('date', inplace=True)
# data_mixed.set_index('date', inplace=True)

# drop covid data ###############################

# data_finance.drop(['1.03.2020', '1.05.2020', '1.04.2020', '1.06.2020'], inplace=True)
# data_macro.drop(['1.03.2020', '1.05.2020', '1.04.2020', '1.06.2020'], inplace=True)
# sentiments.drop(['1.03.2020', '1.05.2020', '1.04.2020', '1.06.2020'], inplace=True)
# data_mixed.drop(['1.03.2020', '1.05.2020', '1.04.2020', '1.06.2020'], inplace=True)
# data_finance_rp.drop(['1.03.2020','1.05.2020','1.04.2020','1.06.2020'],inplace=True)
#######################################################################################

X_values = monthly_dataset_sentiment_92_filtered.shift(1).dropna()

Y_values = monthly_dataset_sentiment_92_filtered['RV30']
Y_values = Y_values.drop(Y_values.index[0])


# X_n=X.drop(['RV30'],axis=1)

# data_finance.drop(['logret, CO1'],axis=1,inplace=True)
# data_finance_rp=data_finance.drop(['logret','CO1','CO3','CO6','brent'], axis=1,inplace=True)

#  Sentiments Regression #########################


#  Sentiments Regression #########################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_values)

# Perform PCA on the scaled data
pca = PCA()
X_reduced = pca.fit_transform(X_scaled)


regr = LinearRegression()
mse = []



#print(np.shape(X_reduced))
# what is the % of explained variance by 1 2 3 ..factors
print('variance explained by factors')
PCA_dim=40
print(np.cumsum(np.round(pca.explained_variance_ratio_, decimals = 4)*100)[0:PCA_dim])



# here we choose the 1st part i.e. 90% for training and 10% for testing,if we want random sample write random

X_reduced_train,X_reduced_test,y_train,y_test = train_test_split(X_reduced,Y_values,test_size=0.02,shuffle=False)

#scale the training and testing data


print(y_test)

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


regresors=get_regresors()



####################### Sentiment only RV30 ##############################################################
#X_train, X_test, y_train, y_test = train_test_split(X_values,Y_values,test_size=0.1,shuffle=False)
rmse_values = []
results, names = list ( ), list ( )

rmse_scores = []

for name, regresor in regresors.items():
    model = regresor
    model.fit(X_reduced_train, y_train)
    y_pred = model.predict(X_reduced_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores.append((name, rmse))
    print(y_test)
    print(y_pred)
    plt.figure (figsize=(10, 4))
    plt.plot (y_test, label='Actual', color='blue', marker='o')
    plt.plot (y_pred, label='Predicted', linestyle='dotted', color='red', marker='o')
    plt.xlabel ('Time')
    plt.ylabel ('Value')
    plt.title (f'Time Series Plot for Model: {name}')
    plt.grid (True)
    plt.xticks (rotation=45)
    plt.legend ( )
    plt.show ( )
# Extract model names and RMSE scores
model_names, rmse_values = zip(*rmse_scores)

# Define colors for bars
colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon','blue']

# Create a bar plot to visualize RMSE for each model
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, rmse_values, color=colors)
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('RMSE for Different Regression Models')

# Add RMSE values as text on each bar
for bar, rmse_val in zip(bars, rmse_values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{rmse_val:.2f}', ha='center', fontsize=10)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(rmse_scores)
df = pd.DataFrame(rmse_scores, columns=['Model', 'RMSE'])

# Save the DataFrame to an Excel file
excel_file_path = 'rmse_scores.xlsx'
df.to_excel(excel_file_path, index=False)

print(f'RMSE scores saved to {excel_file_path}')

