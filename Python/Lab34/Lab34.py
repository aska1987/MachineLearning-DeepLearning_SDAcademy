# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 09:39:52 2018

@author: SDEDU
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pylab as plt

url='http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data=pd.read_csv(url,sep=';')
x=data[['fixed acidity','volatile acidity','citric acid',
       'residual sugar','chlorides','free sulfur dioxide',
       'total sulfur dioxide','density','pH','sulphates',
       'alcohol']]
y=data.quality
data.columns
x=preprocessing.StandardScaler().fit(x).transform(x)
#주성분 분석
model=PCA()
results=model.fit(x)
z=results.transform(x)
plt.plot(results.explained_variance_)


#데이터프레임에서 PCA성분을 보임
pd.DataFrame(results.components_,columns=list(['fixed acidity','volatile acidity','citric acid',
       'residual sugar','chlorides','free sulfur dioxide',
       'total sulfur dioxide','density','pH','sulphates',
       'alcohol']))

## 주성분 분석 이전 포도주 점수 예측 ############################
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

gnb=GaussianNB() #추정을 위해 가우스 분포 나이브 베이즈 분류기 사용
fit=gnb.fit(x,y) #데이터 적합 처리
pred=fit.predict(x) #데이터 예측
confusion_matrix(pred,y) #혼돈 행렬 분석
# ---> 나이브 베이즈 분류기가 1599 중에서 897건을 올바르게 예측 

## Adult 데이터 셋 확인 ######################################
#어떤 근로자의 수입이 50,000 달러를 초과하는지 예측
import os
import mglearn
import pandas as pd
#names 매개변수로 열 이름을 제공
data=pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH,'adult.data'),header=None,
                 index_col=False,names=['age','workclass','fnlwgt','education','education-num','martial-status',
                                        'occupation','relationship','race','gender','capital-gain','capital-loss',
                                        'hours-per-week','native-country','income'])
#예제를 위해 몇개만 선택
data=data[['age','workclass','education','gender','hours-per-week','occupation',
           'income']]
(data.head())
data.gender.value_counts()

data_dummies=pd.get_dummies(data)
data_dummies.columns

features=data_dummies.loc[:,'age':'occupation_ Transport-moving']
x=features.values
y=data_dummies['income_ <=50K'].values
x.shape
y.shape

#성능테스트
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
logreg.score(x_train,y_train) #테스트 점수: 0.81

demo_df=pd.DataFrame({'숫자 특성':[0,1,2,1],'범주형 특성':['양말','여우','양말','상자']})
demo_df

demo_df['숫자 특성']=demo_df['숫자 특성'].astype(str)

print(pd.get_dummies(demo_df,columns=['숫자 특성','범주형 특성']))

## 구간 분할, 이산화, 선형모델,트리모델 ##################
import matplotlib.pyplot as plt
import mglearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

x,y=mglearn.datasets.make_wave(n_samples=100)
line=np.linspace(-3,3,1000,endpoint=False).reshape(-1,1)
reg=DecisionTreeRegressor(min_samples_split=3).fit(x,y)
plt.plot(line,reg.predict(line),label='DecisionTreeRegressor')
reg=LinearRegression().fit(x,y)
plt.plot(line,reg.predict(line),label='LinearRegression')
plt.plot(x[:,0],y,'o',c='k')
plt.xlabel('regressor output')
plt.ylabel('input attr')
plt.legend(loc='best')
#연속형 데이터에 아주 강력한 선형 모델을 만드는 방법은 한 특성을 여러 특성으로 나누는 구간 분할이다(이산화)

bins=np.linspace(-3,3,11) #구간
which_bin=np.digitize(x,bins=bins)
x[:5] #데이터 포인트
which_bin[:5] #데이터 포인트의 소속 구간

from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder(sparse=False)
encoder.fit(which_bin) #which_bin 에 나타난 유일한 값을 찾는다
x_binned=encoder.transform(which_bin) 
x_binned.shape #구간을10개로 정의했기 떄문에 변환된 데이터셋은 10개의 특성으로 구성

## 원핫 인코딩된 데이터로 선형회귀모델과 결정트리모델 ##########
x,y=mglearn.datasets.make_wave(n_samples=100)
line=np.linspace(-3,3,1000,endpoint=False).reshape(-1,1)
bins=np.linspace(-3,3,11) #구간
which_bin=np.digitize(x,bins=bins)
encoder=OneHotEncoder(sparse=False)
encoder.fit(which_bin) #which_bin 에 나타난 유일한 값을 찾는다
x_binned=encoder.transform(which_bin) 
line_binned=encoder.transform(np.digitize(line,bins=bins))
reg=LinearRegression().fit(x_binned,y)
plt.plot(line,reg.predict(line_binned),label='구간 선형 트리')
reg=DecisionTreeRegressor(min_samples_split=3).fit(x_binned,y)
plt.plot(line,reg.predict(line_binned),'--',label='구간 결정 트리')
plt.plot(x[:,0],y,'o',c='k')
plt.xlabel('regressor output')
plt.ylabel('input attr')
plt.legend(loc='best')
'''
구간을 나눔으로써 선형 모델이 훨씬 유연해졌다.
'''
