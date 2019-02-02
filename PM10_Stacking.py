# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:53:18 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
import math
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

n_folds =5
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.7, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='poly', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=3, max_features='sqrt',
                                   min_samples_leaf=12, min_samples_split=30, 
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
#model_xgb = xgb.XGBRegressor()

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
lasso=make_pipeline(RobustScaler(),Lasso(alpha=0.0005,random_state=1))

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
#        temp = out_of_fold_predictions
#        out_of_fold_predictions = preprocessing.scale(np.concatenate((temp, X_train[list(X_train.columns)].values), axis=1))
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
#        temp = meta_features
#        meta_features = preprocessing.scale(np.concatenate((temp, X_test[list(X_test.columns)].values), axis=1))
        return self.meta_model_.predict(meta_features), meta_features


train = pd.read_csv('train.csv')
#trainlabel1 = train['PM10'][0:1500]
#trainlabel2 = train['PM10'][182101:364202]
#train = pd.DataFrame(preprocessing.scale(train.values),columns=train.columns)
#train1 = train[['PM2.5','NO2','CO','O3','SO2']][0:1500]
#train2 = train[['PM2.5','NO2','CO','O3','SO2']][182101:364202]

#train['PM2.5/NO2'] = train['PM2.5']/train['NO2']
#train['CO*O3'] = train['CO']*train['O3']
#train['PM2.5*O3'] = train['PM2.5']*train['O3']
#train['NO2/CO'] = train['NO2']/train['CO']
#train['PM2.5*NO2'] = train['PM2.5']*train['NO2']
#train['NO2*O3'] = train['NO2']*train['O3']
#train.drop(['CO','SO2'],axis=1,inplace=True)
test = pd.read_csv('test.csv')
test=test.fillna(method='bfill')
mytrain = train.dropna()
mytest = test.dropna()

mytrainlabel = mytrain['PM10']
mytrain = mytrain[['aqName', 'time', 'PM2.5', 'NO2', 'CO', 'O3', 'SO2']]
mytest = mytest[['aqName', 'time', 'PM2.5', 'NO2', 'CO', 'O3', 'SO2']]
mydata = pd.concat((mytrain,mytest))
n_mytrain = mytrain.shape[0]
n_mytest = mytest.shape[0]

mydata['year'] = mydata['time'].apply(lambda x:int(x[0:4]))
mydata['month'] = mydata['time'].apply(lambda x:int(x[5:7]))
mydata['day'] = mydata['time'].apply(lambda x:int(x[8:10]))
mydata['hour'] = mydata['time'].apply(lambda x:int(x[-8:-6]))

urbanstation = pd.read_csv('Urban Stations.csv')
urbanstation.dropna(inplace=True)
urbanstation['area'] = 'urbanstation'
suburbanstation = pd.read_csv('Suburban Stations.csv')
suburbanstation.dropna(inplace=True)
suburbanstation['area'] = 'suburbanstation'
otherstation = pd.read_csv('Other Stations.csv')
otherstation.dropna(inplace=True)
otherstation['area'] = 'otherstation'
neartrafficstation = pd.read_csv('Stations Near Traffic.csv')
neartrafficstation.dropna(inplace=True)
neartrafficstation['area'] = 'neartrafficstation'
station = pd.concat((urbanstation,suburbanstation,otherstation,neartrafficstation))
station.columns=['station','longitude','latitude','area']
station = station[['station', 'area']]
station.rename(columns={ station.columns[0]: 'aqName' }, inplace=True)

mydata = pd.merge(mydata, station, on='aqName')

mydata = mydata[['PM2.5', 'NO2', 'CO', 'O3', 'SO2','year', 'month', 'day', 'hour', 'area']]
#mydata = mydata[['PM2.5', 'NO2', 'CO', 'O3', 'SO2']]
#tt = pd.merge(train, station, on='aqName')
#mydata = pd.get_dummies(mydata)
def f(x):
    return {
            'neartrafficstation': 6,
            'urbanstation': 4,
            'suburbanstation': 2,
            'otherstation': 1
            }.get(x, 9)
mydata['area'] = mydata['area'].apply(lambda x:f(x))
mydata = pd.DataFrame(preprocessing.scale(mydata.values),columns=mydata.columns)

mytrain1 = mydata[0:n_mytrain]

mytrainlabel1 = mytrainlabel



mytest1 = mydata[n_mytrain:n_mytrain+n_mytest]

#model_xgb.fit(mytrain1, mytrainlabel1)
#xgb_pred = model_xgb.predict(mytest1)
X_train, X_test, y_train, y_test = train_test_split(mytrain1, mytrainlabel1, test_size=0.9, random_state=42)
#model_xgb.fit(X_train, y_train)
#xgb_pred1 = model_xgb.predict(X_test)
#
#model_lgb.fit(X_train, y_train)
#lgb_pred = model_lgb.predict(X_test)
#mse1 = mean_squared_error(xgb_pred1,y_test)
#mae1 = mean_absolute_error(xgb_pred1,y_test)
#mse = mean_squared_error(lgb_pred,y_test)
#mae = mean_absolute_error(lgb_pred,y_test)

#mytrain1 = mytrain1[['PM2.5', 'NO2', 'CO', 'O3', 'SO2','year', 'month', 'day', 'hour','area']]
#testtrain = mytrain1[0:5000]
#testtrainlabel = mytrainlabel[0:5000]
#X_train, X_test, y_train, y_test = train_test_split(testtrain, testtrainlabel, test_size=0.33, random_state=42)

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),meta_model = lasso)
stacked_averaged_models.fit(X_train.values, y_train.values)
X_test = X_test[0:27543]
stacked_pred,mf = stacked_averaged_models.predict(X_test)
result=stacked_pred
mse = mean_squared_error(result,y_test[0:27543])
mae = mean_absolute_error(result,y_test[0:27543])

#test = pd.DataFrame(preprocessing.scale(test.values),columns=test.columns)
#test = test[['PM2.5','NO2','CO','O3','SO2']]
#test['PM2.5/NO2'] = test['PM2.5']*test['NO2']
#test['CO*O3'] = test['CO']*test['O3']
#test['PM2.5*O3'] = test['PM2.5']*test['O3']
#test['NO2/CO'] = test['NO2']/test['CO']
#test['PM2.5*NO2'] = test['PM2.5']*test['NO2']
#test['NO2*O3'] = test['NO2']*test['O3']
#test.drop(['CO','SO2'],axis=1,inplace=True)
#train.drop('PM10',axis=1,inplace=True)
#train = train[0:5000]
#import copy
#train1 = copy.deepcopy(train)
#train2 = copy.deepcopy(train)
#X_train, X_test, y_train, y_test = train_test_split(train, trainlabel, test_size=0.33, random_state=42)
#stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),meta_model = lasso)
#stacked_averaged_models.fit(train1.values, trainlabel1.values)
#stacked_pred1,mf = stacked_averaged_models.predict(test)
#stacked_averaged_models.fit(train2.values, trainlabel2.values)
#stacked_pred2,mf = stacked_averaged_models.predict(test)

#
#result=stacked_pred1
#mse = mean_squared_error(result,y_test)
#mae = mean_absolute_error(result,y_test)
#dic1 = {}
#dic2 = {}
#for i in range(len(train1.columns)):
#    for j in range(i,len(train1.columns)):
#        a=train1.columns[i]
#        b=train1.columns[j]
#        train = train2
#        train[a+'*'+b] = train[a]*train[b]
#        train = train[0:5000]
#        
#        X_train, X_test, y_train, y_test = train_test_split(train, trainlabel, test_size=0.33, random_state=42)
#        stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),meta_model = lasso)
#        stacked_averaged_models.fit(X_train.values, y_train.values)
#        stacked_pred,mf = stacked_averaged_models.predict(X_test)
#        result=stacked_pred
#        mse = mean_squared_error(result,y_test)
#        mae = mean_absolute_error(result,y_test)
#        print(a+'*'+b+':',mae)
#        dic1[a+'*'+b] = mae
#          
#for i in range(len(train1.columns)):
#    for j in range(len(train1.columns)):
#        a=train1.columns[i]
#        b=train1.columns[j]
#        train = train2
#        train[a+'/'+b] = train[a]/train[b]
#        
#       
#        train = train[0:5000]
#        
#        X_train, X_test, y_train, y_test = train_test_split(train, trainlabel, test_size=0.33, random_state=42)
#        stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),meta_model = lasso)
#        stacked_averaged_models.fit(X_train.values, y_train.values)
#        stacked_pred,mf = stacked_averaged_models.predict(X_test)
#        result=stacked_pred
#        mse = mean_squared_error(result,y_test)
#        mae = mean_absolute_error(result,y_test)
#        print(a+'/'+b+':',mae)
#        dic2[a+'/'+b] = mae
        



#
#model_xgb.fit(X_train, y_train)
#xgb_pred = model_xgb.predict(X_test)
#
#model_lgb.fit(X_train, y_train)
#lgb_pred = model_lgb.predict(X_test.values)
#

##result=stacked_pred*0.7+xgb_pred*0.2+lgb_pred*0.1

#for i in range(len(result)):
#    if result[i]<0:
#        result[i]=0.07
#
#result=pd.DataFrame(result)
#result.columns=['PM10']
#result['id']=result.index
#result=result[['id','PM10']]
#result.to_csv('PM10.csv',index=0)

