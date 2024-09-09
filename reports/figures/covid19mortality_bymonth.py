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

#EDA scatterplot of COVID-19 Death vs Month
month_agg = merged_covid_year_month.groupby("Month").agg(np.sum)["Covid Death"].reset_index()
sns.barplot(month_agg, x="Month", y="Covid Death", color="cornflowerblue")
plt.title("Number of patients deceased from COVID-19 vs Month")
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
plt.show()




