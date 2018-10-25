 # -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 15:46:29 2018

@author: muham
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import SimpleImputer

housing_data = pd.read_csv('train.csv')

# remove columns with data na
housing_data = housing_data.dropna(axis=1)

# separate target and predictors
housing_data_target = housing_data.SalePrice
housing_data_pred = housing_data.drop(['SalePrice'], axis=1)
housing_data_pred = housing_data.select_dtypes(exclude=['object'])

# split the data into test and validation sets
X_train, X_val, y_train, y_val = train_test_split(housing_data_pred,
                                                  housing_data_target,
                                                  train_size=0.7,
                                                  test_size=0.3,
                                                  random_state=0)

# calculate the mae for the different techniques
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    return mean_absolute_error(pred, y_test)

# 1) drop the columns with missing data
cols_with_missing = [col for col in X_train.columns
                         if X_train[col].isnull().any()]
X_train_reduced = X_train.copy()

X_train_reduced = X_train_reduced.drop(cols_with_missing, axis=1)
# 2) impute missing values
imputer = SimpleImputer()
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train))
X_val_imputed = imputer.transform(X_val)
X_train_imputed.columns = X_train.columns
# 3) impute missing values separately
X_train_imputed_plus = X_train.copy()
X_test_imputed_plus = X_val.copy()
cols_with_missing = (col for col in X_train.columns
                     if X_train[col].isnull().any())
for col in cols_with_missing:
    X_train_imputed_plus[col + '_imputed'] = X_train_imputed_plus[col].isnull() 
    X_test_imputed_plus[col + '_imputed'] = X_test_imputed_plus[col].isnull()
    
imputer_plus = SimpleImputer()
X_train_imputed_plus = imputer_plus.fit_transform(X_train_imputed_plus)
X_test_imputed_plus = imputer_plus.transform(X_test_imputed_plus)
    
