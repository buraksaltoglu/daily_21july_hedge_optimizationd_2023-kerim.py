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
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
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
# data_mixed = pd.read_csv(r"data\data_mixed.csv", encoding='utf-8', sep=',')
data=pd.read_csv(r"new_data\dataf.csv", encoding='utf-8', sep=',')
data.set_index('date',inplace=True)
#print(data.head())
data.drop(['2020-01-01','2020-04-01','2020-06-01','2020-05-01'], inplace=True)

X_values = data.shift(1).dropna()
Y_values = data['RV60']
Y_values = Y_values.drop(Y_values.index[0])

# Perform PCA on the scaled data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_values)
pca = PCA()
X_reduced = pca.fit_transform(X_scaled)

# what is the % of explained variance by 1 2 3 ..factors


#mse = []

# what is the % of explained variance by 1 2 3 ..factors
print('variance explained by factors')


cumulative_variance_ratio = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)
index_90_percent = np.argmax(cumulative_variance_ratio >= 90)

# Specify the number of principal components (factors) you want to plot
PCA_dim = len(cumulative_variance_ratio)
explained_variance_90_percent = index_90_percent + 1  # +1 because indexing starts from 0
# Create a plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, PCA_dim + 1), cumulative_variance_ratio[:PCA_dim], marker='o', linestyle='-')
plt.plot(index_90_percent + 1, 90, 'ro')  # Red dot at 90% explained variance
plt.title('Cumulative Explained Variance vs. Number of Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.xticks(range(5, PCA_dim + 1, 5))
#plt.xticks(range(1, PCA_dim + 1), range(1, PCA_dim + 1))  # Show only integer values on the x-axis
plt.grid(True)
plt.show()



print(f"Dimension where 90% variance is explained: {explained_variance_90_percent}")
# here we choose the 1st part i.e. 90% for training and 10% for testing,if we want random sample write random

PCA_factor=index_90_percent
PCA_factor=index_90_percent

X_reduced_train,X_reduced_test,y_train,y_test = train_test_split(X_reduced[:,0:PCA_factor],Y_values,test_size=0.1,shuffle=False)

#scale the training and testing data




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
    regresors['ann'] = MLPRegressor(hidden_layer_sizes=(250,), max_iter=1000)
    regresors[ 'svr' ] = SVR (kernel='linear')  # You can change the kernel type as needed
    regresors[ 'adaboost' ] = AdaBoostRegressor ( )

    return regresors


regresors=get_regresors()

print('----dimension PCA -----')

print(np.shape(X_reduced_train))

rmse_values = []
results, names = list ( ), list ( )
b_plot = pd.DataFrame()
rmse_scores = []
mse_cv_mean_scores= []

scores_cv=pd.DataFrame(columns=['regressor','mse'])
for name, regresor in regresors.items():
    model = regresor
    model.fit(X_reduced_train, y_train)
    y_pred = model.predict(X_reduced_test)
    rmse = np.sqrt(mean_squared_error(y_test.values, y_pred))
    rmse_scores.append((name, rmse))

    r2 = cross_val_score (model, X_reduced_train, y_train, cv=10, scoring='r2')
    cvmse = cross_val_score (model, X_reduced_train, y_train, cv=10, scoring='neg_mean_squared_error')
    scores_cv = scores_cv.append ({'regressor': name, 'cvmse': np.sqrt(-cvmse.mean ( ))}, ignore_index=True)


    # Plot RMSE for each regressor


    #y_test_df = pd.DataFrame(y_test).reset_index()
    #y_pred_df = pd.DataFrame({'Pred': y_pred})
    #y_pred_df['date'] = y_test_df['date']
    #y_all = y_test_df.merge(y_pred_df, on='date')

    scores_cv = pd.DataFrame (scores_cv)

    # we plot fitted and predicted

    #plt.figure (figsize=(10, 4))
    #plt.plot (y_test.values, label='Actual', color='red')
    #plt.plot (y_pred, label='pred', color='blue')
    #plt.xlabel ('Time')
    #plt.ylabel ('Value')
    #plt.title (f'Time Series Plot for Model: {name}')
   # plt.grid (True)
    #plt.xticks (rotation=45)
    #plt.legend ( )
    #plt.show ( )



    b_plot = b_plot.append ({'regressor': name,  'mse': mse}, ignore_index=True)

print(scores_cv)
print(rmse_scores)
# Extract model names and RMSE scores
model_names, rmse_values = zip(*rmse_scores)

print("CV ---mean")

#model mse_cv_mean_scores=zip(*rmse_scores)
# Define colors for bars
colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon','blue']

# Create a bar plot to visualize RMSE for each model
plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, rmse_values, color=colors)
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('RMSE for Different Regression Models')

# Add RMSE values as text on each bar
#for bar, rmse_val in zip(bars, rmse_values):
    #plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{rmse_val:.2f}', ha='center', fontsize=10)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


df = pd.DataFrame(rmse_scores, columns=['Model', 'RMSE'])

# Save the DataFrame to an Excel file
excel_file_path = 'scores_cv.xlsx'
df.to_excel(excel_file_path, index=False)

    print(f'RMSE scores saved to {excel_file_path}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
################################################################
# Initialize empty DataFrames
scores = pd.DataFrame(columns=['regresor', 'rmse'])
b_plot = pd.DataFrame(columns=['regresor', 'rmse'])
validation=pd.DataFrame(columns=['regresor', 'rmse_valid'])



# Lists to store data for boxplot
names_list = []
rmse_mean_list = []
rmse_standard_deviation_list = []
RMSE_cv= []
# List to store data for later plotting
mses_and_corresponding_models = []

for name, regresor in regresors.items():
    model = regresor
    model.fit(X_reduced_train, y_train)
    y_pred = model.predict (X_reduced_test)
    rmse = np.sqrt (mean_squared_error (y_test.values, y_pred))
    r2 = cross_val_score(model, X_reduced_train, y_train, cv=5, scoring='r2')
    mse = cross_val_score(model, X_reduced_train, y_train, cv=5, scoring='neg_mean_squared_error')
    scores = scores.append ( {'regresor': name, 'rmse_stdev': np.std(np.sqrt(-mse)), 'rmse': np.sqrt(-mse.mean())},ignore_index=True)
    b_plot = b_plot.append({'regresor': name, 'rmse_st.dev': np.sqrt(np.var(-mse)), 'rmse': np.sqrt(-mse)}, ignore_index=True)
    validation = validation.append ({'regresor': name, 'rmse_valid':rmse},ignore_index=True)

    #plt.figure (figsize=(10, 4))
    plt.plot (y_test.values, label='Actual', color='red')
    plt.plot (y_pred, label='pred', color='blue')
    plt.xlabel ('Time')
    plt.ylabel ('Value')
    plt.title (f'Time Series Plot for Model: {name}')
    plt.grid (True)
    plt.xticks (rotation=45)
    plt.legend ( )
    #plt.show( )
print('scores')
print(scores)


validation = pd.DataFrame(validation)

# Define the subdirectory path
subdirectory_path = 'cv_mse'

# Create the subdirectory if it doesn't exist
if not os.path.exists(subdirectory_path):
    os.makedirs(subdirectory_path)

# Define file names for scores and validation
scores_file_name = 'scores.xlsx'
validation_file_name = 'validation.xlsx'

# Write the DataFrames to separate Excel files inside the subdirectory
scores_file_path = os.path.join(subdirectory_path, scores_file_name)
validation_file_path = os.path.join(subdirectory_path, validation_file_name)

scores.to_excel(scores_file_path, index=False)
validation.to_excel(validation_file_path, index=False)




validation_rmse=validation['rmse_valid']
regresors = scores['regresor']
rmse = scores['rmse']
rmse_stdev = scores['rmse_stdev']

#rmse_validation=validation['rmse']
colors = [ 'skyblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'blue' ]
plt.title('VALIDATION')
#plt.figure(figsize=(10, 6))
#plt.bar(regresors, validation_rmse, color=colors)
#plt.show()

#validation = pd.DataFrame(validation)
plt.figure(figsize=(10, 6))
plt.bar(validation['regresor'], validation['rmse_valid'], color='skyblue')
plt.xlabel('Regressor')
plt.ylabel('RMSE Valid')
plt.title('Regressor vs. RMSE Valid')

# Print RMSE values on the bars
for i, value in enumerate(validation['rmse_valid']):
    plt.text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=10, color='black')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()

# Create two subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Plot 1: Regressor vs. MSE
axs[0].bar(regresors, rmse, color=colors)
axs[0].set_ylabel('rmse')
axs[0].set_title('Regressor vs. RMSE')

# Plot 2: Regressor vs. MSE Std Dev
axs[1].bar(regresors, rmse_stdev, color=colors)
axs[1].set_ylabel('MSE Std Dev')
axs[1].set_title('Regressor vs. MSE Std Dev')

# Rotate x-axis labels for better readability
for ax in axs:
    ax.tick_params(axis='x', rotation=45)

# Adjust the space between subplots
plt.tight_layout()

# Show the plots
plt.show()