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


#Import data
# change the working directory
by_conditions = pd.read_csv("/Users/dankim/Desktop/Courses/DataC200/Grad_project/cdc_death_counts_by_conditons.csv")
by_others = pd.read_csv("/Users/dankim/Desktop/Courses/DataC200/Grad_project/cdc_death_counts_by_sex_age_state.csv")

#Data processing
#Preprocess data and merge two datasets
by_others["Age Group"] = by_others["Age Group"].str.replace("85 years and over", "85+",regex=True)
by_others["Age Group"] = by_others["Age Group"].str.replace("years", "",regex=True)
by_others["Age Group"] = by_others["Age Group"].str.replace(" ", "",regex=True)
by_others["Age Group"] = by_others["Age Group"].str.replace("Under1year", "0-24",regex=True)
by_others["Age Group"] = by_others["Age Group"].str.replace("0-17", "0-24",regex=True)
by_others["Age Group"] = by_others["Age Group"].str.replace("1-4", "0-24",regex=True)
by_others["Age Group"] = by_others["Age Group"].str.replace("5-14", "0-24",regex=True)
by_others["Age Group"] = by_others["Age Group"].str.replace("15-24", "0-24",regex=True)

#Merge two datasets
merged_covid = pd.merge(by_conditions, by_others, how='inner', on=["Year", "Month","Age Group", "State"])

#Add number of patients deceased from COVID-19
merged_covid["Covid Death"] = merged_covid["COVID-19 Deaths_x"] + merged_covid["COVID-19 Deaths_y"]

#Clean year and month and concatenate, and sex
merged_covid_year_month = merged_covid[~merged_covid["Year"].isna() ]
merged_covid_year_month = merged_covid[~merged_covid["Month"].isna()]
merged_covid_year_month = merged_covid_year_month[merged_covid_year_month["Sex"] != "All Sexes"]
merged_covid_year_month = merged_covid_year_month[merged_covid_year_month["State"] != "United States"]
merged_covid_year_month["Year_Month"] = merged_covid_year_month["Year"].astype(str) + "-" + merged_covid_year_month["Month"].astype(str)


#Linear Ridge regression model on the merged dataset
data = merged_covid_year_month[["Year_Month", "State", "Sex", "Age Group", "Covid Death"]]

#One hot encoding
data = ohe_year_month(data)
data = ohe_state(data)
data = ohe_age_group(data)
data= ohe_sex(data)
data = data.drop(["Year_Month", "State", "Sex", "Age Group"], axis=1)
data = data[~data["Covid Death"].isna()]

#train and test split
train, test = train_test_set_split(data, 0.67)

X_train = train.drop(['Covid Death'], axis=1)
y_train = train["Covid Death"]

X_test = test.drop(["Covid Death"], axis=1)
y_test = test["Covid Death"]

from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [150,250,350],
    'criterion' :['squared_error'],
    'n_estimators': [100,200],
    "min_samples_split" : [10,5,20]
}
# Create a base model
rfCV = RandomForestRegressor()

# Instantiate the grid search model
regCV = GridSearchCV(estimator = rfCV, cv=5,param_grid = param_grid, n_jobs = -1, verbose = 2, return_train_score=True)
# Fit the grid search to the data
model_rf_cv = regCV.fit(X_train, y_train);
pred_train_rf_cv = model_rf_cv.predict(X_train)

print("Random forest regression model score")
print("Training RMSE value: ",np.sqrt(mean_squared_error(y_train,pred_train_rf_cv)))
print("Training R2 score:", r2_score(y_train, pred_train_rf_cv))

pred_test_rf_cv= model_rf_cv.predict(X_test)
print("Test RMSE value: ", np.sqrt(mean_squared_error(y_test,pred_test_rf_cv))) 
print("Test R2 score: ", r2_score(y_test, pred_test_rf_cv))




