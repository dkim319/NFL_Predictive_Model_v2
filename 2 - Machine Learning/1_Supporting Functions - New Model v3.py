# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 22:55:57 2017

@author: DKIM
"""
import pandas as pd
import numpy as np
import datetime
import time

def load_data(start_year, target_year):
    data = pd.read_csv('Data.csv')

    data = data[data['season'] >= start_year]
    
    # replace any null values with 0
    data = data.fillna(0)
    
    # use one-hot coding to replace the favorite and underdog categorical variables
    fav_team = pd.get_dummies(data['favorite'])
    und_team = pd.get_dummies(data['underdog'])
    
    # use a prefix to distinguish the two categorical variables
    fav_team = fav_team.add_prefix('fav_')
    und_team = und_team.add_prefix('und_')
    
    # remove the original fields
    data = data.drop('favorite', axis = 1)
    data = data.drop('underdog', axis = 1)
    
    # add the one-hot coded fields
    data = pd.concat([data, fav_team], axis = 1)
    data = pd.concat([data, und_team], axis = 1)
    
    #print data.head(5)
    #print(data.describe())
    
    # split the dataset into training and testing datasets
    data_train = data[data['season'] <= target_year-1]
    data_train.reset_index()
    data_test = data[data['season'] == target_year]
    data_test.reset_index()
    
    # split training and testing datasets into features and target 
    features_train = data_train.drop('spreadflag', axis = 1)
    target_train = data_train['spreadflag']
    
    features_test = data_test.drop('spreadflag', axis = 1)
    target_test = data_test['spreadflag']
    
    return data_train, data_test, features_train, target_train, features_test, target_test


import xgboost
from sklearn.metrics import accuracy_score

def tune_parameters(features_train, target_train, features_test, target_test):
    # initalize variables
    acc_list_train = [] 
    acc_list_test = []
    
    est_list = []
    lr_list = []
    depth_list = []
    subsample_list = []
    colsamplebt_list = []
    
    n_est = 0
    
    # first iteration
    est = [10,20,30,40,50,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,200,300,400]
    lr = [0.01, 0.1,0.2,0.3]#]
    depth = [4,5,6,7,8,9,10]
    subsample = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
    colsamplebt = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]

    results = pd.DataFrame()
    
    # perform parameter tuning and use the parameters that produce the best results
    for e in est:
        for l in lr:
            for d in depth:
                for s in subsample:
                    for c in colsamplebt:
                        xgb = xgboost.XGBClassifier(n_estimators=e, learning_rate=l, gamma=0, subsample=s,
                                                   colsample_bytree=c, max_depth=d)
                        
                        xgb.fit(features_train, target_train)
                        pred_test = xgb.predict(features_test)
                        pred_train = xgb.predict(features_train)
                        
                        predictions_train = [round(value) for value in pred_train]
                        predictions_test =  [round(value) for value in pred_test]
                        
                        train_accuracy = accuracy_score(target_train, predictions_train)
                        test_accuracy = accuracy_score(target_test, predictions_test)

                        print (train_accuracy)
                        print (test_accuracy)
                        
                        est_list.append(e)
                        lr_list.append(l)   
                        depth_list.append(d)
                        subsample_list.append(s)
                        colsamplebt_list.append(c)
                        
                        acc_list_train.append(train_accuracy)
                        acc_list_test.append(test_accuracy)
    
    results = pd.DataFrame(
            {'est': est_list,
            'lr': lr_list,
            'depth': depth_list,
            'subsample': subsample_list,
            'colsample_bytree': colsamplebt_list,
            'acc_train': acc_list_train,
            'acc_test': acc_list_test
            })
    
    results.to_csv('results.csv')
    
    return results

def feature_select_per(features_train, target_train, features_test, target_test, est, lr, depth, subsample, colsamplebt):
    
    # Now after the model has been tuned, use percentile to do feature selection
    from sklearn import feature_selection
    
    acc_list_train = []
    acc_list_test = []
    
    per_list = []
    
    percentile = range(1,101)
    #range(10,100)
    #percentile = [22]
    
    # identify the percentile that will produce the best results 
    for per in percentile:
        
        # intilaize SelectFromModel using thresh
        fs = feature_selection.SelectPercentile(feature_selection.f_classif, percentile = per)
        feature_model =  fs.fit(features_train,target_train)
    
        features_train_new = feature_model.transform(features_train)
        features_test_new = feature_model.transform(features_test)
    
        xgb = xgboost.XGBClassifier(n_estimators=est, learning_rate=lr, gamma=0, subsample=subsample,
                                   colsample_bytree=colsamplebt, max_depth=depth)
        
        xgb.fit(features_train_new, target_train)
        pred_test = xgb.predict(features_test_new)
        pred_train = xgb.predict(features_train_new)
    
        predictions_train = [round(value) for value in pred_train]
        predictions_test =  [round(value) for value in pred_test]
        
        train_accuracy = accuracy_score(target_train, predictions_train)
        test_accuracy = accuracy_score(target_test, predictions_test)
        
        print (per)
        print (train_accuracy)
        print (test_accuracy)
        
        per_list.append(per)
        acc_list_train.append(train_accuracy)
        acc_list_test.append(test_accuracy)
    
    per_results = pd.DataFrame(
            {'per': per_list,
            'acc_train': acc_list_train,
            'acc_test': acc_list_test
            })
    
    per_results.to_csv('per_results.csv')

    return per_results