# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 10:15:21 2018
iris 데이터 예측하기
forge 데이터셋 이용해서 분류모델 만들기
유방암 데이터셋 이용해서 분류모델
@author: SDEDU
"""
'''
(base) C:\Users\SDEDU>pip install watermark
(base) C:\Users\SDEDU>conda install numpy scipy scikit-learn matplotlib pandas pillow graphviz python-graphviz
'''

from sklearn.datasets import load_iris
iris_dataset=load_iris()

print('{}'.format(iris_dataset.keys()))
print('{}'.format(iris_dataset['target_names']))

print(iris_dataset['feature_names'])
print(iris_dataset['DESCR'][:193]+'\n...')

type(iris_dataset['data'])
iris_dataset['data'].shape
type(iris_dataset['target'])
iris_dataset['target'].shape

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

x_train.shape
y_train.shape
x_test.shape
y_test.shape

import numpy as np
import pandas as pd
import mglearn
iris_dataframe=pd.DataFrame(x_train,columns=iris_dataset.feature_names)

pd.scatter_matrix(iris_dataframe, c=y_train,figsize=(15,15),marker='o',
                  hist_kwds={'bins':20},s=60,alpha=0.8,cmap=mglearn.cm3)

#k-최근접 이웃 알고리즘
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1) # 이웃 1개

#적용
knn.fit(x_train,y_train)
KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',
                     metric_params=None,n_jobs=1,n_neighbors=1,p=2,weights='uniform')

#예측할 데이터
x_new=np.array([[5,2.9,1,0.2]])
x_new.shape

#예측
prediction=knn.predict(x_new)
iris_dataset['target_names'][prediction] #예측
knn.score(x_test,y_test) #정확도

## forge ###########################
import matplotlib.pyplot as plt
import mglearn

x,y=mglearn.datasets.make_forge()
#산점도
mglearn.discrete_scatter(x[:,0],x[:,1],y)
plt.legend(['클래스 0','클래스 1'],loc=4)
plt.xlabel('첫번째 특성')
plt.ylabel('두번쨰 특성')

x.shape

x_train,x_test,y_train,y_test=train_test_split(x,y)
#knn 3개 
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
knn.score(x_test,y_test)

##유방암 데이터 #########################################
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
x=data['data']
y=data['target']
x_train,x_test,y_train,y_test=train_test_split(x,y)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
knn.score(x_test,y_test)
