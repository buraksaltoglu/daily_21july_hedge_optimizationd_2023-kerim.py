# TODO: Modularize Regressor selector
# TODO: Write the best features to excel or csv
# TODO: Save the CSVs to scores directory instead of the root directory

import warnings
import openpyxl
from math import sqrt
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor

sns.set_style('white')
# from finta import TA
# from sklearn.naive_bayes import GaussianN


# import graphviz
warnings.filterwarnings('ignore')

# to see the columns
# for col in data_finance_rp.columns:
# print(col)
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

# getting the ####################################################
data_finance_rp = pd.read_csv(r"data\finance_9_1_2023.csv", encoding='utf-8', sep=';')  # we delete co1,co3
data_finance = pd.read_csv(r"data\finance_6_2_2023.csv", encoding='utf-8', sep=',')  # we delete
data_macro = pd.read_csv(r"data\data_macro_only_13_1_2023.csv", encoding='utf-8', sep=';')
sentiments = pd.read_csv(r"data/sentiments.csv", encoding='utf-8', sep=',')
data_mixed = pd.read_csv(r"data\data_mixed.csv", encoding='utf-8', sep=',')
##############################################################################################################
# create the data index ###################################################
data_finance.set_index("date", inplace=True)
data_finance_rp.set_index('date', inplace=True)
data_macro.set_index('date', inplace=True)
sentiments.set_index('date', inplace=True)
data_mixed.set_index('date', inplace=True)

# drop covid data ###############################

data_finance.drop(['1.03.2020', '1.05.2020', '1.04.2020', '1.06.2020'], inplace=True)
data_macro.drop(['1.03.2020', '1.05.2020', '1.04.2020', '1.06.2020'], inplace=True)
sentiments.drop(['1.03.2020', '1.05.2020', '1.04.2020', '1.06.2020'], inplace=True)
data_mixed.drop(['1.03.2020', '1.05.2020', '1.04.2020', '1.06.2020'], inplace=True)
# data_finance_rp.drop(['1.03.2020','1.05.2020','1.04.2020','1.06.2020'],inplace=True)
#######################################################################################

X_values = sentiments.shift(1).dropna()

Y_values = sentiments['RV30']
Y_values = Y_values.drop(Y_values.index[0])

# X_n=X.drop(['RV30'],axis=1)

# data_finance.drop(['logret, CO1'],axis=1,inplace=True)
# data_finance_rp=data_finance.drop(['logret','CO1','CO3','CO6','brent'], axis=1,inplace=True)

#  Sentiments Regression #########################


def get_regressors():
    regressors_dict = dict()
    regressors_dict['cart'] = DecisionTreeRegressor(max_depth=5)
    regressors_dict['ols'] = LinearRegression()
    regressors_dict['rforest'] = RandomForestRegressor(max_depth=2, random_state=1)
    regressors_dict["gboost"] = GradientBoostingRegressor()
    regressors_dict['XGBoost'] = XGBRegressor()
    regressors_dict['LGBoost'] = LGBMRegressor()
    regressors_dict['ridge'] = Ridge(alpha=1.0)
    regressors_dict['Lasso'] = Lasso(alpha=0.0005)
    return regressors_dict


def evaluate_regressor(regressor, x_n, y):
    selector = RFE(regressor, n_features_to_select=5, step=1)
    selector = selector.fit(x_n, y)

    return selector


def take_square_root(mse_array):
    for i in range(len(mse_array)):
        mse_array[i] = sqrt(abs(mse_array[i]))


regressors = get_regressors()
results, names = list(), list()


"""
for name, regressor in regressors.items():
    selector = evaluate_regressor(regressor=regressor, x_n=X_values, y=Y_values)
    results.append(selector)
    names.append(name)
    print(name, selector.support_)
    print(name, selector.ranking_)
    best_features = set()
    for i in range(len(selector.ranking_)):
        if selector.ranking_[i] == 1:
            print("index of feature with ranking 1")
            print(i)
            print(sentiments.columns[i])  # TODO: Burada çıkan RV30 aslında RV30'un lagi
            best_features.add(sentiments.columns[i])
    print("\n")
f = open("best_features.txt", "a")

for e in best_features:
    f.write(e)
    f.write("\n")
f.close()
"""

def get_features_of_dataset(data_set):
    feature_set = []
    for j in range(len(data_set.columns)):
        feature_set.append([data_set.columns[j]])
    return feature_set


print("Test for getting the feature set")

ml_methods = ["Feature Names"]
for key in regressors.keys():
    ml_methods.append(key)
feature_set_sentiments = [ml_methods] + get_features_of_dataset(data_set=sentiments)
# TODO: Pad with zeros
for i in range(1, len(feature_set_sentiments)):
    for k in range(len(feature_set_sentiments[0]) - 1):
        feature_set_sentiments[i].append(0)
print("test for feature finding")
index_of_current_feature = 1
for name, regressor in regressors.items():
    selector = evaluate_regressor(regressor=regressor, x_n=X_values, y=Y_values)
    results.append(selector)
    names.append(name)
    print()
    print(name, selector.support_)
    print(name, selector.ranking_)

    index_of_current_feature = feature_set_sentiments[0].index(name)
    for i in range(1, len(selector.ranking_) + 1):
        feature_set_sentiments[i][index_of_current_feature] = selector.ranking_[i - 1]


for e in feature_set_sentiments:
    print(e)
    print("\n")

# Open a new workbook
workbook = openpyxl.Workbook()

# Select the active worksheet (the first one by default)
worksheet = workbook.active

# Write the matrix values to the Excel worksheet
for row_index, row_data in enumerate(feature_set_sentiments, start=1):
    for col_index, value in enumerate(row_data, start=1):
        worksheet.cell(row=row_index, column=col_index, value=value)

# Save the workbook to a file
path_to_feature_selection_directory = r"feature_selection\ "

path_to_feature_selection_directory = re.sub(r'\s+', '', path_to_feature_selection_directory)
file_name = "feature_set.xlsx"
path_to_feature_selection_directory += file_name
workbook.save(path_to_feature_selection_directory)

def plotter(data_set, frequency: str, data_type: str, r_mse: bool):
    """
    :param data_set: Data set such as sentiments or data_macro
    :param frequency: Frequency of the data, such as 30, 60 or 90
    :param data_type: Name of main feature of the data set such as Sentiment or Mixed Data
    :param r_mse: If r_mse is true we take the square root og the mse values. Otherwise, we return the mse values.
    :return:
    """

    ##################################################

    rv_name = "RV" + frequency
    plot_title = rv_name + " with " + data_type + " Data"
    if r_mse is True:
        plot_title = "RMSE: " + plot_title
    else:
        plot_title = "MSE: " + plot_title

    x = data_set.shift(1).dropna()
    y = data_set[rv_name]
    y = y.drop(y.index[0])  # TODO: Do we have to drop index 0 for all of them?

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.80, shuffle=False)
    b_plot = pd.DataFrame()
    scores = pd.DataFrame(columns=['regressor', 'r2', 'mse', 'RV'])

    for name, regressor in regressors.items():
        model = regressor
        model.fit(x_train, y_train)
        r2 = cross_val_score(model, x, y, cv=5, scoring='r2')
        mse = cross_val_score(model, x, y, cv=5, scoring='neg_mean_squared_error')

        if r_mse is True:
            take_square_root(mse_array=mse)

            scores = scores.append(
                {'regressor': name, 'r2': r2.mean(), 'mse': mse.mean(), 'RV': rv_name}, ignore_index=True)
            b_plot = b_plot.append({'regressor': name, 'r2': r2, 'mse': mse, 'RV': rv_name}, ignore_index=True)
        else:
            scores = scores.append(
                {'regressor': name, 'r2': r2.mean(), 'mse': -mse.mean(), 'RV': rv_name}, ignore_index=True)
            b_plot = b_plot.append({'regressor': name, 'r2': r2, 'mse': mse, 'RV': rv_name}, ignore_index=True)

    if r_mse is True:
        plt.boxplot(b_plot.mse, labels=regressors.keys(), showmeans=True, notch=True, patch_artist=True)
    else:
        plt.boxplot(-b_plot.mse, labels=regressors.keys(), showmeans=True, notch=True, patch_artist=True)

    plt.title(plot_title)
    plt.show()


###########################################################################################################
# Example usage: plotter(data_set=sentiments, frequency="30", data_type="Sentiment", r_mse=False)

# plotter(data_set=sentiments, frequency="30", data_type="Sentiment", r_mse=False)
# plotter(data_set=sentiments, frequency="60", data_type="Sentiment", r_mse=False)
# plotter(data_set=sentiments, frequency="90", data_type="Sentiment", r_mse=False)

# plotter(data_set=data_finance, frequency="30", data_type="Finance", r_mse=False)
# plotter(data_set=data_finance, frequency="60", data_type="Finance", r_mse=False)
# plotter(data_set=data_finance, frequency="90", data_type="Finance", r_mse=False)

# plotter(data_set=data_macro, frequency="30", data_type="Macro", r_mse=False)
# plotter(data_set=data_macro, frequency="60", data_type="Macro", r_mse=False)
# plotter(data_set=data_macro, frequency="90", data_type="Macro", r_mse=False)
