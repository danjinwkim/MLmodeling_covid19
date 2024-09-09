import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model as lm
import zipfile
import os
import csv
import re
from sklearn import linear_model as lm

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder


def train_test_set_split(data, percentage):
    data_len = data.shape[0]
    shuffled_indices = np.random.permutation(data_len)
    proportion = int(percentage*data_len)
    training = shuffled_indices[0:proportion]
    test = shuffled_indices[proportion:]
    return data.iloc[training], data.iloc[test]

def process_data_gm(data, pipeline_functions, prediction_col):
    """Process the data for a guided model."""
    for function, arguments, keyword_arguments in pipeline_functions:
        if keyword_arguments and (not arguments):
            data = data.pipe(function, **keyword_arguments)
        elif (not keyword_arguments) and (arguments):
            data = data.pipe(function, *arguments)
        else:
            data = data.pipe(function)
    X = data.drop(columns=[prediction_col]).to_numpy()
    y = data.loc[:, prediction_col].to_numpy()
    return X, y

def ohe_year_month(data):
    """
    One-hot-encodes roof material.  New columns are of the form x0_MATERIAL.
    """
    ...
    # BEGIN SOLUTION NO PROMPT
    oh_enc = OneHotEncoder()
    oh_enc.fit(data[['Year_Month']])
    dummies = pd.DataFrame(oh_enc.transform(data[['Year_Month']]).todense(),
                           columns=oh_enc.get_feature_names_out(),
                           index = data.index)
    return data.join(dummies)

def ohe_state(data):
    """
    One-hot-encodes roof material.  New columns are of the form x0_MATERIAL.
    """
    ...
    # BEGIN SOLUTION NO PROMPT
    oh_enc = OneHotEncoder()
    oh_enc.fit(data[['State']])
    dummies = pd.DataFrame(oh_enc.transform(data[['State']]).todense(),
                           columns=oh_enc.get_feature_names_out(),
                           index = data.index)
    return data.join(dummies)


def ohe_sex(data):
    """
    One-hot-encodes roof material.  New columns are of the form x0_MATERIAL.
    """
    ...
    # BEGIN SOLUTION NO PROMPT
    oh_enc = OneHotEncoder()
    oh_enc.fit(data[['Sex']])
    dummies = pd.DataFrame(oh_enc.transform(data[['Sex']]).todense(),
                           columns=oh_enc.get_feature_names_out(),
                           index = data.index)
    return data.join(dummies)


def ohe_age_group(data):
    """
    One-hot-encodes roof material.  New columns are of the form x0_MATERIAL.
    """
    ...
    # BEGIN SOLUTION NO PROMPT
    oh_enc = OneHotEncoder(handle_unknown = 'ignore')
    oh_enc.fit(data[['Age Group']])
    dummies = pd.DataFrame(oh_enc.transform(data[['Age Group']]).todense(),
                           columns=oh_enc.get_feature_names_out(),
                           index = data.index)
    return data.join(dummies)

def ohe_condition(data):
    """
    One-hot-encodes roof material.  New columns are of the form x0_MATERIAL.
    """
    ...
    # BEGIN SOLUTION NO PROMPT
    oh_enc = OneHotEncoder()
    oh_enc.fit(data[['Condition']])
    dummies = pd.DataFrame(oh_enc.transform(data[['Condition']]).todense(),
                           columns=oh_enc.get_feature_names_out(),
                           index = data.index)
    return data.join(dummies)


def ohe_condition_group(data):
    """
    One-hot-encodes roof material.  New columns are of the form x0_MATERIAL.
    """
    ...
    # BEGIN SOLUTION NO PROMPT
    oh_enc = OneHotEncoder()
    oh_enc.fit(data[['Condition Group']])
    dummies = pd.DataFrame(oh_enc.transform(data[['Condition Group']]).todense(),
                           columns=oh_enc.get_feature_names_out(),
                           index = data.index)
    return data.join(dummies)


def ohe_month(data):
    """
    One-hot-encodes roof material.  New columns are of the form x0_MATERIAL.
    """
    ...
    # BEGIN SOLUTION NO PROMPT
    oh_enc = OneHotEncoder()
    oh_enc.fit(data[['Month']])
    dummies = pd.DataFrame(oh_enc.transform(data[['Month']]).todense(),
                           columns=oh_enc.get_feature_names_out(),
                           index = data.index)
    return data.join(dummies)

def drop_columns(data, *columns):
    """Select only columns passed as arguments."""
    return data.loc[:, columns]

def rmse(predicted, actual):
    """
    Calculates RMSE from actual and predicted values
    Input:
      predicted (1D array): vector of predicted/fitted values
      actual (1D array): vector of actual values
    Output:
      a float, the root-mean square error
    """
    return np.sqrt(np.mean((actual - predicted)**2))

def data_no_yearmonth(data):
    data = data[["Age Group", "Sex", "State", "Covid Death"]]

    #One hot encoding
    #data = ohe_year_month(data)
    data = ohe_state(data)
    data = ohe_age_group(data)
    data = ohe_sex(data)
    data = data.drop(["Age Group",'Sex', "State"], axis=1)
    data = data[~data["Covid Death"].isna()]
    return data


def data_no_state(data):
    data = data[["Year_Month", "Sex", "Age Group", "Covid Death"]]

    #One hot encoding
    data = ohe_year_month(data)
    #data = ohe_state(data)
    data = ohe_age_group(data)
    data = ohe_sex(data)
    data = data.drop(["Year_Month",'Sex', "Age Group"], axis=1)
    data = data[~data["Covid Death"].isna()]
    return data

def data_no_sex(data):
    data = data[["Year_Month", "State", "Age Group", "Covid Death"]]

    #One hot encoding
    data = ohe_year_month(data)
    data = ohe_state(data)
    data = ohe_age_group(data)
    #data_no_sex= ohe_sex(data_no_sex)
    data = data.drop(["Year_Month",'State', "Age Group"], axis=1)
    data = data[~data["Covid Death"].isna()]
    return data

def data_no_agegroup(data):
    data = data[["Year_Month", "Sex", "State", "Covid Death"]]

    #One hot encoding
    data = ohe_year_month(data)
    data = ohe_state(data)
    #data = ohe_age_group(data)
    data= ohe_sex(data)
    data = data.drop(["Year_Month",'Sex', "State"], axis=1)
    data = data[~data["Covid Death"].isna()]
    return data