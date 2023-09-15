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

data_macro = file_dictionary['monthly_dataset_macro_sentiment_92_filtered']
monthly_dataset_sentiment_92_filtered = file_dictionary['monthly_dataset_sentiment_92_filtered']
sentiments=monthly_dataset_sentiment_92_filtered
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
column_names = data_macro.columns

# Now, 'column_names' contains the names of all columns in the DataFrame 'df'
print(column_names)

X_values = sentiments.shift(1).dropna()

Y_values = sentiments['RV30']
Y_values = Y_values.drop(Y_values.index[0])

# X_n=X.drop(['RV30'],axis=1)

# data_finance.drop(['logret, CO1'],axis=1,inplace=True)
# data_finance_rp=data_finance.drop(['logret','CO1','CO3','CO6','brent'], axis=1,inplace=True)

#  Sentiments Regression #########################


def get_regressors():
    regressors_dict = dict()
    regressors_dict['cart'] =    DecisionTreeRegressor(max_depth=5)
    regressors_dict['ols'] =     LinearRegression()
    regressors_dict['rforest'] = RandomForestRegressor(max_depth=2, random_state=1)
    regressors_dict["gboost"] =  GradientBoostingRegressor()
    regressors_dict['XGBoost'] = XGBRegressor()
    regressors_dict['LGBoost'] = LGBMRegressor()
    regressors_dict['ridge'] =   Ridge(alpha=1.0)
    regressors_dict['Lasso'] =   Lasso(alpha=0.0005)
    return regressors_dict


def evaluate_regressor(regressor, x_n, y, n_features_to_select):
    selector = RFE(regressor, n_features_to_select=n_features_to_select, step=1)
    selector = selector.fit(x_n, y)

    return selector


def take_square_root(mse_array):
    for i in range(len(mse_array)):
        mse_array[i] = sqrt(abs(mse_array[i]))


regressors = get_regressors()
results, names = list(), list()

def write_to_csv(data, file_name: str, directory: str):
    file_path = r"" + directory + "\ "
    file_path = re.sub(r'\s+', '', file_path)
    file_path += file_name

    with open(file_path, 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the data from the list to the CSV file
        csv_writer.writerows(data)

def write_matrix_to_excel(matrix, directory_name: str, file_name: str):
    """
    :param matrix: Matrix formed by python nested lists
    :param directory_name: The directory that we want to save the file
    :param file_name: The name of the file
    :return:
    """
    workbook = openpyxl.Workbook()

    # Select the active worksheet (the first one by default)
    worksheet = workbook.active

    # Write the matrix values to the Excel worksheet
    for row_index, row_data in enumerate(matrix, start=1):
        for col_index, value in enumerate(row_data, start=1):
            worksheet.cell(row=row_index, column=col_index, value=value)

    # Get the path to directory
    path_to_directory = r""
    path_to_directory += directory_name + "\ "
    path_to_directory = re.sub(r'\s+', '', path_to_directory)

    # Convert the empty spaces to underscores
    file_name = re.sub(r'\s', '_', file_name)

    #  Add the file name and extension to the path
    path_to_directory = path_to_directory + file_name + ".xlsx"

    # Save the file
    workbook.save(path_to_directory)


def get_features_of_dataset(data_set):
    """
    This function assumes that the 'date' feature has beem dropped from the data_set
    so, the first feature is a meaningful one
    :param data_set: Data set that has the features and observations.
    :return: Returns the list of features in this data set.
    """
    feature_set = []
    for j in range(len(data_set.columns)):
        feature_set.append([data_set.columns[j]])
    return feature_set


def sfs_backward(data_set, target: str, n_features: int, scoring: str, file_name: str, directory: str):
    x_train, x_test, y_train, y_test = train_test_split(data_set, data_set[target], random_state=0, train_size=.90)
    feature_names = np.array(get_features_of_dataset(data_set))
    regressors = get_regressors()
    feature_matrix = []
    # evaluate the models and store results
    names_of_methods = []
    for name, regressor in regressors.items():
        sfs_backward = SequentialFeatureSelector(
            regressor, n_features_to_select=n_features, scoring=scoring, direction="backward", n_jobs=-1).fit(x_test, y_test)
        # print(names.append(name))
        names_of_methods.append(regressor)
        print(regressor)
        # print(sfs_backward.get_support())
        feature_matrix.append(feature_names[sfs_backward.get_support()])
        print((feature_names[sfs_backward.get_support()]))
        # print(results.append(feature_names[sfs_backward.get_support()]))
        # print(feature_names[sfs_backward.get_support()])

    (
        "Features selected by backward sequential selection: "
        f"{feature_names[ sfs_backward.get_support ( ) ]}"
    )
    for e in feature_matrix:
        print(e)
        print("\n")
    for j in range(len(feature_matrix)):
        feature_matrix[j] = feature_matrix[j].tolist()

    for j in range(len(feature_matrix)):
        feature_matrix[j].insert(0, names_of_methods[j])

    write_to_csv(data=feature_matrix, file_name=file_name, directory=directory)

def feature_set_selector(data_set, target: str, regressors_for_data_set: dict, n_features_to_select: int, directory_name:str, file_name: str):
    """
    :param data_set: Data set such as sentiments or data_macro
    :param target: Independent variable
    :param regressors_for_data_set: Regressor for the data set. See 'def get_regressors()'.
    :param n_features_to_select: First n best features
    :param file_name: The name of the file that the results will be saved to. The file extension is xlsx by default
    :param directory_name: The name of the directory that the results will be saved to.
    :return:
    """

    ml_methods = ["Feature Names"]
    # Get the names of the different regressors
    for key in regressors_for_data_set.keys():
        ml_methods.append(key)
    feature_selection_index_matrix = [ml_methods] + get_features_of_dataset(data_set=data_set)

    # Pad the matrix with zeros
    for i in range(1, len(feature_selection_index_matrix)):
        for k in range(len(feature_selection_index_matrix[0]) - 1):
            feature_selection_index_matrix[i].append(0)

    ###############################################################################################
    print("test for feature finding")
    for name, regressor in regressors_for_data_set.items():
        selector = evaluate_regressor(regressor=regressor, x_n=data_set, y=data_set[target],
                                      n_features_to_select=n_features_to_select)
        results.append(selector)
        names.append(name)
        print()
        print(name, selector.support_)  # TODO: Delete the print statements
        print(name, selector.ranking_)

        index_of_current_feature = feature_selection_index_matrix[0].index(name)
        for i in range(1, len(selector.ranking_) + 1):
            feature_selection_index_matrix[i][index_of_current_feature] = selector.ranking_[i - 1]

    # Write the feature_selection_index_matrix_to_excel
    write_matrix_to_excel(matrix=feature_selection_index_matrix, directory_name=directory_name, file_name=file_name)
    return feature_selection_index_matrix


def update_data_set(data_set, threshold: int, target: str, regressors_for_data_set: dict, n_features_to_select: int,
                    directory_name: str, file_name: str):
    # TODO: Finish this function
    # Remove the features that are below a certain threshold.
    # We can see the performance of features from the matrix returned from --> def feature_set_selector()
    # Update the data_set by removing the features that are below the threshold
    # Return the updated data set

    performance_matrix_for_features = feature_set_selector(data_set=data_set, target=target,
                                                           regressors_for_data_set=regressors_for_data_set,
                                                           n_features_to_select=n_features_to_select,
                                                           directory_name=directory_name, file_name=file_name)
    print("Length of performance matrix")
    print(len(performance_matrix_for_features))
    print("Length of performance matrix[0]")
    print(len(performance_matrix_for_features[0]))

    print("Performance matrix")
    for e in performance_matrix_for_features:
        print(e)
        print("\n")

    features_to_be_removed = []
    for j in range(1, len(performance_matrix_for_features)):
        this_feature_is_below_threshold = False
        for k in range(1, len(performance_matrix_for_features[j])):
            if performance_matrix_for_features[j][k] > threshold:
                this_feature_is_below_threshold = True
                break

        if this_feature_is_below_threshold is True:
            features_to_be_removed.append(performance_matrix_for_features[j][0])

    updated_data_set = copy.deepcopy(data_set)

    print("features to be removed")
    print(features_to_be_removed)

    print("updated data set before")
    print(updated_data_set)
    for feature in features_to_be_removed:
        if feature != "date":
            updated_data_set.pop(feature)

    print("\n")
    print("updated data set after")
    print(updated_data_set)

    print("TYPE")
    print(type(updated_data_set))
    # TODO: Write to excel does not work
    # updated_data_set acts like a dictionary
    print("date test")
    print(updated_data_set.keys())
    # features = updated_data_set.keys().tolist()
    # updated_data_set = updated_data_set.values.tolist()
    # updated_data_set.insert(0, features)
    # write_matrix_to_excel(matrix=updated_data_set, directory_name=directory_name, file_name=file_name)
    updated_data_set.to_csv("macro")
    return updated_data_set


def plotter(data_set, regressors_for_data_set, frequency: str, data_type: str, r_mse: bool, directory_name: str, file_name: str):
    """
    :param data_set: Data set such as sentiments or data_macro
    :param regressors_for_data_set: Regressor for the data set. See 'def get_regressors()'.
    :param frequency: Frequency of the data, such as 30, 60 or 90
    :param data_type: Name of main feature of the data set such as Sentiment or Mixed Data
    :param r_mse: If r_mse is true we take the square root og the mse values. Otherwise, we return the mse values
    :param directory_name: The name of the directory that we want to save our results
    :param file_name: The name of the file that we want to save our results in. The default extension of the file is xlsx.
    :return:
    """
    ##################################################

    rv_name = "RV" + frequency  # rv stands for realized volatility

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

    names_list = ["ML Methods"]
    mse_mean_list = ["Mean"]
    mse_standard_deviation_list = ["Standard Deviation"]
    mses_and_corresponding_models = []

    for name, regressor in regressors_for_data_set.items():
        model = regressor
        model.fit(x_train, y_train)
        r2 = cross_val_score(model, x, y, cv=5, scoring='r2')
        mse = cross_val_score(model, x, y, cv=5, scoring='neg_mean_squared_error')

        if r_mse is True:
            take_square_root(mse_array=mse)

            scores = scores.append(
                {'regressor': name, 'r2': r2.mean(), 'mse': mse.mean(), 'RV': rv_name}, ignore_index=True)
            b_plot = b_plot.append({'regressor': name, 'r2': r2, 'mse': mse, 'RV': rv_name}, ignore_index=True)
            names_list.append(name)
            mse_mean_list.append(mse.mean())
            mse_standard_deviation_list.append(mse.std())
        else:
            scores = scores.append(
                {'regressor': name, 'r2': r2.mean(), 'mse': -mse.mean(), 'RV': rv_name}, ignore_index=True)
            b_plot = b_plot.append({'regressor': name, 'r2': r2, 'mse': mse, 'RV': rv_name}, ignore_index=True)
            names_list.append(name)
            mse_mean_list.append(-mse.mean())
            mse_standard_deviation_list.append(mse.std())

    mses_and_corresponding_models.append(names_list)
    mses_and_corresponding_models.append(mse_mean_list)
    mses_and_corresponding_models.append(mse_standard_deviation_list)

    if r_mse is True:
        plt.boxplot(b_plot.mse, labels=regressors.keys(), showmeans=True, notch=True, patch_artist=True)
    else:
        plt.boxplot(-b_plot.mse, labels=regressors.keys(), showmeans=True, notch=True, patch_artist=True)

    # Plot the mse graph
    plt.title(plot_title)
    plt.show()

    # Save the mses and corresponding models to excel
    write_matrix_to_excel(matrix=mses_and_corresponding_models, directory_name=directory_name, file_name=file_name)
    # End of function


###########################################################################################################
# Example usage for plotter is below:
"""
plotter(data_set=data_macro, regressors_for_data_set=regressors, frequency="30", data_type="Macro", r_mse=True,
        directory_name="forecast_performance", file_name="RV30 RMSE Data Macro")

plotter(data_set=data_macro, regressors_for_data_set=regressors, frequency="60", data_type="Macro", r_mse=True,
        directory_name="forecast_performance", file_name="RV60 RMSE Data Macro")

plotter(data_set=data_macro, regressors_for_data_set=regressors, frequency="90", data_type="Macro", r_mse=True,
        directory_name="forecast_performance", file_name="RV90 RMSE Data Macro")
"""

# Example usage for feature_set_selector is below:

"""
feature_set_selector(data_set=sentiments, target="RV30", regressors_for_data_set=regressors, n_features_to_select=3,
                     directory_name="feature_selection", file_name="sentiments_feature_set_30")
feature_set_selector(data_set=sentiments, target="RV60", regressors_for_data_set=regressors, n_features_to_select=3,
                     directory_name="feature_selection", file_name="sentiments_feature_set_60")
feature_set_selector(data_set=sentiments, target="RV90", regressors_for_data_set=regressors, n_features_to_select=3,
                     directory_name="feature_selection", file_name="sentiments_feature_set_90")
"""

# Example usage for update_data_set is below:
update_data_set(data_set=sentiments, target="RV60", regressors_for_data_set=regressors, n_features_to_select=5,
                directory_name="updated_data_sets", file_name="updated_sentiments_feature_set_60", threshold=7)

# update_data_set(data_set=data_macro, target="RV30", regressors_for_data_set=regressors, n_features_to_select=15,
                # directory_name="updated_data_sets", file_name="updated_macro_feature_set_30", threshold=30)

# Example usage for feature_set_selector is below:
sfs_backward(data_set=data_macro, target="RV60", scoring="r2", n_features=5,
             file_name="sentiments_RV30_features_sfs_backward_RV30_features_sfs_backward", directory="sfs_backward_feature_selection")

