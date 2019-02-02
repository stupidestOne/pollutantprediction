# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 10:39:10 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import SGDRegressor

n_folds =5
par = PassiveAggressiveRegressor()
sgd = SGDRegressor()
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
                
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),meta_model = lasso)
#stacked_averaged_models = StackingAveragedModels(base_models = (model_lgb, sgd),meta_model = par)
train = pd.read_csv('totalTrain.csv')
test_weather = pd.read_csv('totalTest.csv')

mytrain = train.dropna()
mytest = test_weather.dropna()

mytrainlabel = mytrain[['PM2.5','PM10','O3']]

mytrain = mytrain[['aqName', 'time', 'temperature','pressure', 'humidity', 'wind_direction', 'wind_speed', 'weather']]
mytest = mytest[['aqName', 'time', 'temperature','pressure', 'humidity', 'wind_direction', 'wind_speed', 'weather']]
#mytest = mytest[['aqName', 'time', 'PM2.5', 'NO2', 'CO', 'O3', 'SO2']]
mydata = pd.concat((mytrain,mytest))
n_mytrain = mytrain.shape[0]
n_mytest = mytest.shape[0]
#
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

mydata = mydata[['temperature','pressure', 'humidity', 'wind_direction', 'wind_speed', 'weather','area','year','month','day','hour']]
def pressure(x):
    if x>1100 or x<900:
        return 0
    else:
        return x
mydata['pressure'] = mydata['pressure'].apply(lambda x:pressure(x))
mydata['pressure'] = mydata['pressure'].replace(0,mydata['pressure'].mean())

def temperature(x):
    if x>50 or x<-30:
        return np.NaN
    else:
        return x
mydata['temperature'] = mydata['temperature'].apply(lambda x:temperature(x))
mydata['temperature'] = mydata['temperature'].replace(np.NaN,mydata['temperature'].mean())
def humidity(x):
    if x>100 or x<0:
        return np.NaN
    else:
        return x
mydata['humidity'] = mydata['humidity'].apply(lambda x:humidity(x))
mydata['humidity'] = mydata['humidity'].replace(np.NaN,mydata['humidity'].mean())
def f(x):
    return {
            'neartrafficstation': 6,
            'urbanstation': 4,
            'suburbanstation': 2,
            'otherstation': 1
            }.get(x)
mydata['area'] = mydata['area'].apply(lambda x:f(x))
def weekday(x,y,z):
    today = datetime.datetime(x,y,z)
    return today.weekday()
mydata['weekday']  = mydata.apply(lambda x:weekday(x['year'],x['month'],x['day']),axis=1)#0 is monday,6 is sunday
def windspeed(x):
    if x>=0 and x<=0.2:
        return 0
    elif x>=0.3 and x<=1.5:
        return 1
    elif x>=1.6 and x<=3.3:
        return 2
    elif x>=3.4 and x<=5.4:
        return 3
    elif x>=5.5 and x<=7.9:
        return 4
    elif x>=8.0 and x<=10.7:
        return 5
    elif x>=10.8 and x<=13.8:
        return 6
    elif x>=13.9 and x<=17.1:
        return 7
    elif x>=17.2 and x<=20.7:
        return 8
    elif x>=20.8 and x<=24.4:
        return 9
    elif x>=24.5 and x<=28.4:
        return 10
    elif x>=28.5 and x<=32.6:
        return 11
    elif x>=32.7 and x<=36.9:
        return 12
    elif x>=37.0 and x<=61.2:
        return 13
    else:
        return np.NaN
mydata['wind_speed'] = mydata['wind_speed'].apply(lambda x:windspeed(x))
mydata['wind_speed'] = mydata['wind_speed'].replace(np.NaN,mydata['wind_speed'].mean())
#
#


#for i in list(mydata.columns):
#    mydata[i]=stats.zscore(mydata[i])
lbl = preprocessing.LabelEncoder()
lbl.fit(list(mydata['weather'].values))
mydata['weather'] = lbl.transform(list(mydata['weather'].values))
#mydata = pd.get_dummies(mydata)
mydata = pd.DataFrame(preprocessing.scale(mydata.values),columns=mydata.columns)

mytrain1 = mydata[0:n_mytrain]

mytrainlabel1 = np.log1p(mytrainlabel)
#mytrainlabel1 = mytrainlabel['PM10']
#mytrainlabel1 = mytrainlabel['PM2.5']
#mytrainlabel1 = mytrainlabel['O3']

mytest1 = mydata[n_mytrain:n_mytrain+n_mytest]


#def smape(actual, predicted):
#    dividend= np.abs(np.array(actual) - np.array(predicted))
#    c = np.array(actual) + np.array(predicted)
#    return 2 * np.mean(np.divide(dividend, c , out=np.zeros_like(dividend), where=c !=0, casting='unsafe'))

#X_train, X_test, y_train, y_test = train_test_split(mytrain1, mytrainlabel1, test_size=0.33, random_state=42)
#stacked_averaged_models.fit(X_train.values, y_train.values)
#
#stacked_pred = stacked_averaged_models.predict(X_test)
#
#model_xgb.fit(X_train, y_train)
#model_xgb_pred = model_xgb.predict(X_test)
#model_lgb.fit(X_train, y_train)
#model_lgb_pred = model_xgb.predict(X_test)
#result=stacked_pred
#mae = mean_absolute_error(stacked_pred,y_test)
#smape = smape(stacked_pred,y_test)

stacked_averaged_models.fit(mytrain1.values, mytrainlabel1.values)
result = stacked_averaged_models.predict(mytest1)
result = np.expm1(result)
result=pd.DataFrame(result)
result.columns=['PM10']
result['id']=result.index
result=result[['id','PM10']]
result.to_csv('PM10.csv',index=0)

#result.columns=['PM2.5']
#result['id']=result.index
#result=result[['id','PM2.5']]
#result.to_csv('PM2.5.csv',index=0)
#result.columns=['O3']
#result['id']=result.index
#result=result[['id','O3']]
#result.to_csv('O3.csv',index=0)
#
#urb_st = list(set(list(urbanstation['Station ID'])))
#urb_st=[i[0:-3] for i in urb_st]
#sub_st = list(set(list(suburbanstation['Station ID'])))
#sub_st=[i[0:-3] for i in sub_st]

#data[data['wind_direction']>=360]['wind_direction'].value_counts()

#X_train, X_test, y_train, y_test = train_test_split(mytrain1, mytrainlabel1, test_size=0.1, random_state=42)
#def iter_minibatches(mytrain,trainlabel,minibatch_size=5000):
#    X = []
#    y = []
#    cur_line_num = 0
#    for i in range(mytrain.shape[0]):
#        y.append(float(trainlabel.iloc[i]))
#        X.append(mytrain.iloc[i]) 
#
#        cur_line_num += 1
#        if cur_line_num >= minibatch_size:
#            X, y = np.array(X), np.array(y)  # 将数据转成numpy的array类型并返回
#            yield X, y
#            X, y = [], []
#            cur_line_num = 0
#from sklearn.linear_model import SGDRegressor
#sgd_reg = SGDRegressor()
#minibatch_train_iterators = iter_minibatches(X_train,y_train,minibatch_size=5000)
#
#for i, (x1, y1) in enumerate(minibatch_train_iterators):
#    sgd_reg.partial_fit(x1, y1)
#    print("{} time".format(i))  # 当前次数
#    print("{} score".format(mean_absolute_error(sgd_reg.predict(X_test), y_test)))  # 在测试集上看效果

#from sklearn.linear_model import PassiveAggressiveRegressor
#par = PassiveAggressiveRegressor()
#minibatch_train_iterators = iter_minibatches(X_train,y_train,minibatch_size=5000)
#
#for i, (x1, y1) in enumerate(minibatch_train_iterators):
#    par.partial_fit(x1, y1)
#    print("{} time".format(i))  # 当前次数
#    print("{} score".format(mean_absolute_error(par.predict(X_test), y_test)))  # 在测试集上看效果
