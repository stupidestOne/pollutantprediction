#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[5]:


import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[16]:


train=pd.re ad_csv('5002data/enhanced/totalTrain.csv')


# In[47]:


aqName=train['aqName'].unique().tolist()
road='5002data/aqFiles/'
end='.csv'
for i in range(len(aqName)):
    train.loc[train['aqName']==aqName[i]].to_csv(road+aqName[i]+end,index=0)


# In[17]:


aqName=train['aqName'].unique().tolist()
road='5002data/aqFiles/'
end='.csv'


# In[18]:


def buildTrain(train, pastHour=48, futureHour=48):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-futureHour-pastHour):
        X_train.append(np.array(train.iloc[i:i+pastHour]))
        Y_train.append(np.array(train.iloc[i+pastHour:i+pastHour+futureHour][['PM2.5','PM10','O3']]))
    return np.array(X_train), np.array(Y_train)


# In[19]:


def augFeatures(train):
    train['time'] = pd.to_datetime(train['time'])
    train['year'] = train['time'].dt.year
    train['month'] = train['time'].dt.month
    train['day'] = train['time'].dt.day
    train['hour'] = train['time'].dt.hour
    return train


# In[20]:


def shuffle(X,Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]


# In[21]:


def splitData(X,Y,rate):
    X_train = X[int(X.shape[0]*rate):]
    Y_train = Y[int(Y.shape[0]*rate):]
    X_val = X[:int(X.shape[0]*rate)]
    Y_val = Y[:int(Y.shape[0]*rate)]
    return X_train, Y_train, X_val, Y_val


# In[22]:


def buildModel(shape):
    model = Sequential()
    model.add(LSTM(16, input_length=shape[1], input_dim=shape[2], return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(TimeDistributed(Dense(3)))
    model.compile(loss="mae", optimizer="adam")
    model.summary()
    return model


# In[ ]:


for i in range(18,len(aqName)):
    now=pd.read_csv(road+aqName[i]+end)
    now=now.dropna()
    now.drop(['aqName','longitude','latitude','obsName','gridName','flag','NO2','CO','SO2'],axis=1,inplace=True)
    now=augFeatures(now)
    now.drop('time',axis=1,inplace=True)
    #print(now.head(2))
    weatherMap={label:idx for idx,label in enumerate(np.unique(now['weather']))}
    now['weather']=now['weather'].map(weatherMap)
    regNowFeatures= now[['temperature','pressure','humidity','wind_direction','wind_speed','weather','year','month','day','hour']].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    regNow=pd.concat([regNowFeatures,now[['PM2.5','PM10','O3']]],axis=1)
    testNow=[]
    testNow.append(np.array(regNow.iloc[regNow.shape[0]-48:regNow.shape[0]]))
    X_train, Y_train = buildTrain(regNow)
    X_train, Y_train = shuffle(X_train, Y_train)
    X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)
    model = buildModel(X_train.shape)
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])
    
    result=model.predict(np.array(testNow))
    result=result[0].tolist()
    for j in range(len(result)):
        name=aqName[i]+'#'+str(j)
        name=[name]
        result[j]=name+result[j]
    result=pd.DataFrame(result)
    result.columns=['test_id','PM2.5','PM10','O3']
    result.to_csv('5002data/result/'+aqName[i]+end,index=0)


# In[ ]:




