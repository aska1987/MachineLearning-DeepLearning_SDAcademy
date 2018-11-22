# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 14:39:58 2018
랜덤포레스트 
citibike 데이터로 예측
international-airline-passengers 데이터로 예측
@author: SDEDU
"""

## 특정 날짜와 시간에 앤디 집 앞에 있는 자전거를 사람들이 얼마나 대여할 것인지를 예측
import pylab as plt
import mglearn
import pandas as pd
citibike=mglearn.datasets.load_citibike()

plt.figure(figsize=(10,3))
xticks=pd.date_range(start=citibike.index.min(),end=citibike.index.max(),freq='D')
week=['sun','mon','tue','wed','thu','fri','sat']
xticks_name=[week[int(w)]+d for w,d in zip(xticks.strftime('%w'),xticks.strftime('%m-%d'))]

plt.xticks(xticks,xticks_name,rotation=90,ha='left')
plt.plot(citibike,linewidth=1)
plt.xlabel('date')
plt.ylabel('rent cnt')

## POSIX 시간만 사용하여 만든 랜덤 포레스트의 예측 ####################
y=citibike.values

#POSIX 시간을 10**9로 나누어 반환
x=citibike.index.astype('int64').values.reshape(-1,1)
#처음 184개 데이터 포인트를 훈련세트로 사용
n_train=184
def eval_on_features(features,target,regressor):
    x_train,x_test=features[:n_train],features[n_train:]
    y_train,y_test=target[:n_train],target[n_train:]
    regressor.fit(x_train,y_train)
    print('테스트세트R^2:{:.2f}'.format(regressor.score(x_test,y_test)))
    y_pred=regressor.predict(x_test)
    y_pred_train=regressor.predict(x_train)
    plt.figure(figsize=(10,3))
    plt.xticks(range(0,len(x),8),xticks_name,
               rotation=90,ha='left')
    plt.plot(range(n_train),y_train,label='train')
    plt.plot(range(n_train,len(y_test)+n_train),y_test,'-',label='test')
    plt.plot(range(n_train),y_pred_train,'--',label='predict train')
    plt.plot(range(n_train,len(y_test)+n_train),y_pred,'--',label='predict test')
    plt.legend(loc=(1.01,0))
    plt.xlabel('date')
    plt.ylabel('rent cnt')
    
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100,random_state=0)
eval_on_features(x,y,regressor)
'''
랜덤포레스트
훈련세트의 예측은 매우 정확
하지만 테스트 세트에 대해선 한가지 값으로만 예측
R^2은 -0.04로 거의 아무것도 학습되지않음


훈련 데이터의 그래프를 보면 시간과 요일이라는 두 요소가 중요
이 두 특성을 추가, POSIX 시간으로는 아무것도 학습되지 않으므로 이 특성은 제외
시간만 사용하여 만든 랜덤 포레스트 예측
'''
x_hour=citibike.index.hour.values.reshape(-1,1)
eval_on_features(x_hour,y,regressor)
#훨씬 나아졋지만 주간패턴은 예측하지 못하는것 같다

#요일 추가
import numpy as np
x_hour_week=np.hstack([citibike.index.dayofweek.values.reshape(-1,1),
                       citibike.index.hour.values.reshape(-1,1)])
eval_on_features(x_hour_week,y,regressor)
#하루의 시간과 요일에 따른 주기적인 패턴을 따르고 있음
#R^2은 0.84로 상당히 좋은 성능

from sklearn.linear_model import LinearRegression
eval_on_features(x_hour_week,y,LinearRegression())
#성능 최악, 패턴도 이상 ( 요일과 시간이 정수로 인코딩 되어 있어서 연속형 변수로 해석되기 때문)
#선형 모델은 시간을 선형함수로만 학습할 수 있어서 하루에서 시간이 흐를수록 대여수가 늘어나게 학습되었다.

#OneHotEncoder를 사용해서 정수형을 범주형 변수로 해석
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
x_hour_week_onehot=enc.fit_transform(x_hour_week).toarray()
eval_on_features(x_hour_week_onehot,y,Ridge())
#연속형 특성일때보다 성능이 좋아짐
#요일에 대해 하나의 계수를 학습하고 시간에 대해서도 하나의 계수를 학습
#이 말은 시간패턴이 모든 날에 걸쳐 공유된다는 뜻

## international-airline-passengers 데이터 실습 ##################
data=pd.read_csv('international-airline-passengers.csv')
data.isnull().sum()
data=data.dropna()
data.Month=pd.to_datetime(data['Month'],format='%Y-%m')
data.index=data.Month
del data['Month']

#xticks=pd.date_range(start=data.index.min(),end=data.index.max(),freq='M')
xticks=pd.date_range(start=data.index.min(),end=data.index.max(),freq='Y')

#xticks_name=[d for d in zip(xticks.strftime('%Y-%m'))]
xticks_name=[d for d in zip(xticks.strftime('%Y'))]

plt.xticks(xticks,xticks_name,rotation=60,ha='left')
plt.plot(data,linewidth=1)
plt.xlabel('date')
plt.ylabel('rent cnt')


y=data.values
x=data.index.astype('int64').values.reshape(-1,1)
n_train=108


def eval_on_features_2(features,target,regressor):
    x_train,x_test=features[:n_train],features[n_train:]
    y_train,y_test=target[:n_train],target[n_train:]
    regressor.fit(x_train,y_train)
    print('테스트세트R^2:{:.2f}'.format(regressor.score(x_test,y_test)))
    y_pred=regressor.predict(x_test)
    y_pred_train=regressor.predict(x_train)
    plt.figure(figsize=(10,3))
    plt.xticks(range(0,len(x),12),xticks_name,
               rotation=60,ha='left')
    plt.plot(range(n_train),y_train,label='train')
    plt.plot(range(n_train,len(y_test)+n_train),y_test,'-',label='test')
    plt.plot(range(n_train),y_pred_train,'--',label='predict train')
    plt.plot(range(n_train,len(y_test)+n_train),y_pred,'--',label='predict test')
    plt.legend(loc=(1.01,0))
    plt.xlabel('date')
    plt.ylabel('rent cnt')

regressor=RandomForestRegressor(n_estimators=50,random_state=0)
eval_on_features_2(x,y,regressor)

x_hour=data.index.year.values.reshape(-1,1)
eval_on_features_2(x_hour,y,regressor)

x_hour_week=np.hstack([data.index.month.values.reshape(-1,1),
                       data.index.year.values.reshape(-1,1)])
eval_on_features_2(x_hour_week,y,regressor)

eval_on_features_2(x_hour_week,y,LinearRegression())

enc=OneHotEncoder()
x_hour_week_onehot=enc.fit_transform(x_hour_week).toarray()
eval_on_features_2(x_hour_week_onehot,y,Ridge())
