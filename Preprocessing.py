#!/usr/bin/env python
# coding: utf-8

# # data exploration&pre-processing

# In[1]:


import pandas as pd
import math
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


grid17=pd.read_csv('5002data/gridWeather_201701-201803.csv')
grid18=pd.read_csv('5002data/gridWeather_201804.csv')
observed17=pd.read_csv('5002data/observedWeather_201701-201801.csv')
observed18_1=pd.read_csv('5002data/observedWeather_201802-201803.csv')
observed18_2=pd.read_csv('5002data/observedWeather_201804.csv')


# In[3]:


aQ17=pd.read_csv('5002data/airQuality_201701-201801.csv')
aQ18_1=pd.read_csv('5002data/airQuality_201802-201803.csv')
aQ18_2=pd.read_csv('5002data/airQuality_201804.csv')


# In[4]:


aQStation=pd.read_csv('5002data/Beijing_AirQuality_Stations.csv')


# In[5]:


aQ17=aQ17.rename(columns={'stationId':'aqName','utc_time':'time'})
aQ18_1=aQ18_1.rename(columns={'stationId':'aqName','utc_time':'time'})
aQ18_2=aQ18_2.rename(columns={'station_id':'aqName','PM25_Concentration':'PM2.5','PM10_Concentration':'PM10','NO2_Concentration':'NO2','CO_Concentration':'CO','O3_Concentration':'O3','SO2_Concentration':'SO2'})
aQ18_2.drop('id',axis=1,inplace=True)


# In[6]:


totalAQ=pd.concat([aQ17,aQ18_1,aQ18_2])


# In[7]:


#sns.countplot(totalAQ['PM10'].isna())


# In[8]:


import missingno as msno
msno.matrix(totalAQ,figsize=(12,5))#可视化查询缺失值


# In[9]:


totalAQ.loc[(totalAQ['PM10'].isnull().values==True)&(totalAQ['PM2.5'].isnull().values==True)&             (totalAQ['NO2'].isnull().values==True)&(totalAQ['CO'].isnull().values==True)&             (totalAQ['O3'].isnull().values==True)&(totalAQ['SO2'].isnull().values==True)].shape


# In[10]:


totalAQ2=totalAQ[(True^(totalAQ['PM10'].isnull().values&totalAQ['PM2.5'].isnull().values                       &totalAQ['NO2'].isnull().values&totalAQ['CO'].isnull().values
                       &totalAQ['O3'].isnull().values&totalAQ['SO2'].isnull().values))]


# In[11]:


totalAQ2.shape
totalAQ2=totalAQ2.reset_index(drop=True)


# In[12]:


corrmat=totalAQ2.corr()
plt.subplots()
sns.heatmap(corrmat,vmax=0.9,square=True)


# In[13]:


fig, ax = plt.subplots()
ax.scatter(x = totalAQ2['aqName'], y = totalAQ2['PM2.5'])
plt.ylabel('PM2.5', fontsize=13)
plt.xlabel('station', fontsize=13)
plt.show()


# In[14]:


totalAQ2=totalAQ2.loc[totalAQ2['PM2.5']<=1400]


# In[15]:


test=totalAQ2.loc[totalAQ2['PM10'].isna()].drop(['PM10'],axis=1)


# In[16]:


train=totalAQ2[(True^totalAQ2['PM10'].isnull().values)]


# In[17]:


label=train['PM10']
features=train.drop(['PM10','aqName','time'],axis=1)


# In[18]:


test=test.fillna(method='bfill')
features=features.fillna(method='bfill')


# In[19]:


#from scipy.stats import zscore  
#for feature in features:
#    features[feature]=zscore(features[feature])


# In[20]:


import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[60]:


n_folds =5

def mae_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(features.values)
    mae= -cross_val_score(model, features.values, label, scoring="neg_mean_absolute_error", cv = kf)
    return(mae)


# In[61]:


model_lgb = lgb.LGBMRegressor()
score=mae_cv(model_lgb)       
print(score.mean(),score.std())


# In[23]:


model_lgb.fit(features, label)
pred=model_lgb.predict(test.drop(['aqName','time'],axis=1).values)
test['PM10']=pred
test=test[['aqName','time','PM2.5','PM10','NO2','CO','O3','SO2']]
totalAQ3=train.append(test).sort_index()


# In[24]:


totalAQ3['PM10']=round(totalAQ3['PM10'])
totalAQ3=totalAQ3.fillna(method='bfill')
totalAQ3.to_csv('5002data/enhanced/totalAirQuaility.csv')


# In[25]:


def round_up(value):
    return round(value * 10) / 10.0


# In[26]:


#to get the distance between the target and the gridweather and observedweather
aQStation['powGridDistance']=pow((aQStation['longitude']-round_up(aQStation['longitude'])),2)+pow((aQStation['latitude']-round_up(aQStation['latitude'])),2)
site=observed17[['station_id','longitude','latitude']]
site=site.drop_duplicates()
site=site.reset_index()
site.drop('index',axis=1, inplace=True)

siteLong=site['longitude']
siteLati=site['latitude']
siteName=site['station_id']
aQLong=aQStation['longitude']
aQLati=aQStation['latitude']
distance=[]
name=[]
for i in range(len(aQLong)):
    dis=[]
    for j in range(len(siteLong)):
        d=pow((aQLong[i]-siteLong[j]),2)+pow((aQLati[i]-siteLati[j]),2)
        dis.append(d)
    distance.append(min(dis))
    name.append(siteName[dis.index(min(dis))])
    
aQStation['powObsDistance']=distance
aQStation['obsName']=name

aQStation['gridLong']=round_up(aQStation['longitude'])
aQStation['gridLati']=round_up(aQStation['latitude'])


gridSite=grid17[['stationName','longitude','latitude']]
gridSite=gridSite.drop_duplicates().reset_index(drop=True)
gridSite=gridSite.rename(columns={'stationName':'gridName','longitude':'gridLong','latitude':'gridLati'})

aQStation=pd.merge(aQStation,gridSite,how='left')

aQStation['flag']=aQStation['powGridDistance']-aQStation['powObsDistance']
aQStation=aQStation.rename(columns={'Station ID':'aqName'}).drop(['powGridDistance','powObsDistance','gridLong','gridLati'],axis=1)
#flag<0 grid近 flag>0 obs近
aQStation.to_csv('5002data/enhanced/newAQStation.csv',index=0)


# In[27]:


aQStation.head(2)


# In[28]:


observed17=observed17.rename(columns={'station_id':'obsName','utc_time':'time'}).drop(['longitude','latitude'],axis=1)
observed18_1=observed18_1.rename(columns={'station_id':'obsName','utc_time':'time'})
observed18_1=observed18_1[['obsName','time','temperature','pressure','humidity','wind_direction','wind_speed','weather']]
observed18_2=observed18_2.rename(columns={'station_id':'obsName'}).drop('id',axis=1)
observed18_2=observed18_2[['obsName','time','temperature','pressure','humidity','wind_direction','wind_speed','weather']]


# In[29]:


totalOW=pd.concat([observed17,observed18_1,observed18_2]).reset_index(drop=True)


# In[30]:


grid17=grid17.rename(columns={'stationName':'gridName','utc_time':'time','wind_speed/kph':'wind_speed'}).drop(['longitude','latitude'],axis=1)
grid17['wind_speed']=grid17['wind_speed']*5/18
grid17['weather']=np.nan
grid18=grid18.rename(columns={'station_id':'gridName'}).drop('id',axis=1)
grid18['wind_speed']=grid18['wind_speed']*5/18
grid18=grid18[['gridName','time','temperature','pressure','humidity','wind_direction','wind_speed','weather']]
#grid17从2017-01-01 00:00:00到2018-03-27 05:00:00 都没有weather


# In[31]:


totalGW=pd.concat([grid17,grid18]).reset_index(drop=True)


# In[32]:


weatherMap={'CLEAR_DAY':'Sunny/clear','HAZE':'Haze','PARTLY_CLOUDY_DAY':'Partly Cloudy',            'WIND':'Wind','CLOUDY':'Cloudy','CLEAR_NIGHT':'Sunny/clear',            'PARTLY_CLOUDY_NIGHT':'Partly Cloudy','RAIN':'Rain','SNOW':'Snow'}


# In[33]:


totalGW['weather']=totalGW['weather'].map(weatherMap)


# In[34]:


totalGW['weather'].unique()


# In[35]:


totalOW['weather'].unique()


# In[36]:


def cutDirec(direc):
    if 0<=direc<45:
        return 0
    elif 45<=direc<90:
        return 1
    elif 90<=direc<135:
        return 2
    elif 135<=direc<180:
        return 3
    elif 180<=direc<225:
        return 4
    elif 225<=direc<270:
        return 5
    elif 270<=direc<315:
        return 6
    elif 315<=direc<=360:
        return 7
    else:
        return -1


# In[37]:


totalOW.loc[totalOW['wind_speed']<0.5,'wind_direction']=-1
totalGW.loc[totalGW['wind_speed']<0.5,'wind_direction']=-1
totalOW['wind_direction']=totalOW['wind_direction'].map(cutDirec)
totalGW['wind_direction']=round(totalGW['wind_direction']).map(cutDirec)


# In[38]:


totalGW.head(5)


# In[39]:


totalGW.to_csv('5002data/enhanced/totalGW.csv')
totalOW.to_csv('5002data/enhanced/totalOW.csv')


# In[40]:


totalTrain=pd.merge(totalAQ3,aQStation,how='left')
totalTrain=pd.merge(totalTrain,totalOW,how='left')


# In[41]:


totalTrain=totalTrain.drop_duplicates()
totalTrain.shape


# In[42]:


gridTotalTrain0=totalTrain.loc[totalTrain['time']<'2017-01-30 16:00:00']
gridTotalTrain1=totalTrain.loc[(totalTrain['flag']<0)&(totalTrain['time']<='2018-03-27 05:00:00')&(totalTrain['time']>='2017-01-30 16:00:00')]
gridTotalTrain2=totalTrain.loc[(totalTrain['flag']<0)&(totalTrain['time']>='2018-04-01 00:00:00')]
gT0=gridTotalTrain0.drop(['temperature','pressure','humidity','wind_direction','wind_speed'],axis=1)
gT1=gridTotalTrain1.drop(['temperature','pressure','humidity','wind_direction','wind_speed'],axis=1)
gT2=gridTotalTrain2.drop(['temperature','pressure','humidity','wind_direction','wind_speed','weather'],axis=1)


# In[43]:


gT2.shape


# In[44]:


gT0=pd.merge(gT0,totalGW.drop('weather',axis=1),how='left')
gT1=pd.merge(gT1,totalGW.drop('weather',axis=1),how='left')
gT2=pd.merge(gT2,totalGW,how='left')


# In[45]:


weather=gT1['weather']
gT1.drop('weather',axis=1,inplace=True)
gT1['weather']=weather
weather=gT0['weather']
gT0.drop('weather',axis=1,inplace=True)
gT0['weather']=weather


# In[46]:


totalTrain.shape


# In[47]:


totalTrain=totalTrain.append(gridTotalTrain0).append(gridTotalTrain1).append(gridTotalTrain2)
totalTrain=totalTrain.drop_duplicates(keep=False)
totalTrain.shape


# In[48]:


totalTrain=totalTrain.append(gT0).append(gT1).append(gT2)
totalTrain=totalTrain.sort_values(by=['aqName','time']).reset_index(drop=True)
totalTrain.shape


# In[49]:


ow=totalTrain.loc[(totalTrain['pressure'].isnull().values==True)&(totalTrain['flag']<0)]
ow2=ow.drop(['temperature','pressure','humidity','wind_direction','wind_speed','weather'],axis=1)
gw=totalTrain.loc[(totalTrain['pressure'].isnull().values==True)&(totalTrain['flag']>0)]
gw2=gw.drop(['temperature','pressure','humidity','wind_direction','wind_speed','weather'],axis=1)
ow2=pd.merge(ow2,totalOW,how='left')
gw2=pd.merge(gw2,totalGW,how='left')


# In[50]:


totalTrain=totalTrain.append(ow).append(gw)
totalTrain=totalTrain.drop_duplicates(keep=False)
totalTrain.shape


# In[51]:


totalTrain=totalTrain.append(ow2).append(gw2)
totalTrain=totalTrain.sort_values(by=['aqName','time']).reset_index(drop=True)
totalTrain.shape


# In[54]:


gw=totalTrain.loc[totalTrain['wind_speed'].isna().values==True]
others=totalTrain.append(gw).drop_duplicates(keep=False)
gw=gw.drop(['temperature','pressure','humidity','wind_direction','wind_speed'],axis=1)
gw=pd.merge(gw,totalGW.drop('weather',axis=1),how='left')
gw=gw[['aqName','time','PM2.5','PM10','NO2','CO','O3','SO2','longitude','latitude','obsName','gridName','flag','temperature','pressure','humidity','wind_direction','wind_speed','weather']]


# In[55]:


totalTrain=others.append(gw).sort_values(by=['aqName','time']).reset_index(drop=True)


# In[58]:


totalTrain.to_csv('5002data/enhanced/totalTrain.csv',index=0)


# In[591]:


from scipy.stats import norm, skew
from scipy import stats
sns.distplot(totalTrain['PM2.5'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(totalTrain['PM2.5'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('pollution distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(totalTrain['PM2.5'], plot=plt)
plt.show()


# In[427]:


fig, ax = plt.subplots()
ax.scatter(x = totalTrain['aqName'], y = totalTrain['PM2.5'])
plt.ylabel('PM2.5', fontsize=13)
plt.xlabel('station', fontsize=13)
plt.show()


# In[428]:


totalTrain.loc[totalTrain['PM2.5']>1400]


# In[200]:


gridTarget=pd.read_csv('5002data/gridWeather_20180501-20180502.csv')
obsTarget=pd.read_csv('5002data/observedWeather_20180501-20180502.csv')


# In[201]:


gridTarget['weather']=gridTarget['weather'].map(weatherMap)


# In[202]:


gridTarget.drop(['id'],axis=1,inplace=True)
gridTarget=gridTarget.rename(columns={'station_id':'gridName'})
obsTarget.drop(['id'],axis=1,inplace=True)
obsTarget=obsTarget.rename(columns={'station_id':'obsName'})


# In[203]:


gridTarget['wind_speed']=gridTarget['wind_speed']*5/18
gridTarget.loc[gridTarget['wind_speed']<0.5,'wind_direction']=-1
obsTarget.loc[obsTarget['wind_speed']<0.5,'wind_direction']=-1
obsTarget['wind_direction']=obsTarget['wind_direction'].map(cutDirec)
gridTarget['wind_direction']=round(gridTarget['wind_direction']).map(cutDirec)


# In[204]:


time=pd.DataFrame(gridTarget['time']).drop_duplicates().values.tolist()


# In[205]:


aqName=aQStation['aqName'].values.tolist()


# In[206]:


target=[]
for i in range(len(aqName)):
    for j in range(len(time)):
        data=[aqName[i],time[j][0]]
        target.append(data)


# In[207]:


target=pd.DataFrame(target)
target.columns=['aqName','time']


# In[210]:


target=pd.merge(target, aQStation, how='left')


# In[211]:


target=pd.merge(target,obsTarget,how='left')
target.shape


# In[212]:


gTarget=target.loc[target['flag']<0]
other=target.append(gTarget).drop_duplicates(keep=False)
gTarget.drop(['weather','temperature','pressure','humidity','wind_speed','wind_direction'],axis=1,inplace=True)
gTarget=pd.merge(gTarget,gridTarget,how='left')
target=other.append(gTarget).sort_values(by=['aqName','time']).reset_index(drop=True)


# In[220]:


gTarget2=target.loc[(target['flag']>0)&(target['temperature'].isnull().values==True)]
other=target.append(gTarget2).drop_duplicates(keep=False)
gTarget2.drop(['weather','temperature','pressure','humidity','wind_speed','wind_direction'],axis=1,inplace=True)
gTarget2=pd.merge(gTarget2,gridTarget,how='left')
target=other.append(gTarget2).sort_values(by=['aqName','time']).reset_index(drop=True)


# In[223]:


target=target[['aqName','time','longitude','latitude','obsName','gridName','flag','temperature','pressure','humidity','wind_direction','wind_speed','weather']]


# In[224]:


target.to_csv('5002data/enhanced/totalTest.csv')

