# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 14:35:56 2018
폐암 수술 환자의 생존율 예측
@author: SDEDU
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import 

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
##1. 환자의 패턴도 그래프로 나타내기 ##############################
data=pd.read_csv('thoraric_surgery.csv',header=None)
data.isnull().sum()
x=data.drop([17],axis=1)
y=data[17]

(data.groupby(17).mean()).plot.bar(rot=0)


data.hist()
data.boxplot()

data[17].value_counts().plot(kind='bar')

data.corr()

x=preprocessing.StandardScaler().fit(x).transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y)

#주성분 분석
model=PCA()
results=model.fit(x)
z=results.transform(x)
plt.plot(results.explained_variance_)


#특성 중요도 시각화
data.shape[1]
feature_names=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
def plot_feature_importances_cancer(model):
    n_features=x.shape[1]
    plt.barh(range(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),feature_names)
    plt.xlabel('attr importances')
    plt.ylabel('attr')
    plt.ylim(-1,n_features)
    plt.legend(loc='best')
plot_feature_importances_cancer(forest)



## 2. 머신러닝 학습, 새로운 환자 예측 ##############################
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

data=pd.read_csv('thoraric_surgery.csv',header=None)

x=data.drop([17],axis=1)
y=data[17]
x_train,x_test,y_train,y_test=train_test_split(x,y)

#randomForest 분류
forest=RandomForestClassifier()
forest.fit(x_train,y_train)
forest.score(x_train,y_train) #0.98 ,0.96
forest.score(x_test,y_test) #0.86


knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
knn.score(x_test,y_test) #0.84, 0.85



