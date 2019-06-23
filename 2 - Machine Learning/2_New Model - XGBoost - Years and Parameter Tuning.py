# -*- coding: utf-8 -*-
"""

@author: DKIM
"""

#Run Supporting functions first
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import feature_selection


seednumber = 319

# load data

start_year = 2014
target_year = 2018

data_train, data_test, features_train, target_train, features_test, target_test = load_data(start_year, target_year)

# start timer to capture the amount of time taken to process this python file
from datetime import datetime
timestart = datetime.now()

# tune the parameters
pt_results = tune_parameters(features_train, target_train, features_test, target_test)
pt_results = pt_results.sort_values(by='acc_test', ascending=False)
pt_results = pt_results.reset_index()

# select the parameters with the highest testing accuracy
est = pt_results['est'][0]
lr = pt_results['lr'][0]
depth = pt_results['depth'][0]
subsample = pt_results['subsample'][0]
colsample_bytree = pt_results['colsample_bytree'][0]

# feature select based on percentile
fs_results = feature_select_per(features_train, target_train, features_test, target_test, est, lr, depth, subsample, colsample_bytree)
fs_results = fs_results.sort_values(by='acc_test', ascending=False).reset_index()

# select the best percentile
per = fs_results['per'][0]

# re-run the model with the best parameters and features selected
fs = feature_selection.SelectPercentile(feature_selection.f_classif, percentile = per)
feature_model =  fs.fit(features_train,target_train)

features_train_new = feature_model.transform(features_train)
features_test_new = feature_model.transform(features_test)

# Create the model
xgb = xgboost.XGBClassifier(n_estimators=est, learning_rate=lr, gamma=0, subsample=subsample,
                           colsample_bytree=colsample_bytree, max_depth=depth)

# Fit the model
xgb.fit(features_train_new, target_train)
pred_test = xgb.predict(features_test_new)
pred_train = xgb.predict(features_train_new)

# predict the games
predictions_train = [round(value) for value in pred_train]
predictions_test =  [round(value) for value in pred_test]

# calculate the accuracy
train_accuracy = accuracy_score(target_train, predictions_train)
test_accuracy = accuracy_score(target_test, predictions_test)

print (train_accuracy)
print (test_accuracy)

# store the predictions
pred_df1 = pd.DataFrame(predictions_test) 
pred_df1_raw = pd.DataFrame(pred_test)

data_test = data_test.reset_index()
pred_df1 = pred_df1.reset_index()
pred_df1_raw = pred_df1_raw.reset_index()

# print out results
data_test['predictions - Percentile'] = pred_df1[0]
data_test.to_csv('predictions for - ' + str(per) + ' - ' + str(start_year) + ' - ' + str(target_year) +'.csv')   

# stop the timer and print out the duration
print (datetime.now() - timestart)