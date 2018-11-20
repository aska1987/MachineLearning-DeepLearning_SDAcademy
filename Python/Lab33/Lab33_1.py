# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:49:39 2018
K-Mean 군집(클러스터)
@author: SDEDU
"""

import mglearn
mglearn.plots.plot_kmeans_algorithm()


mglearn.plots.plot_kmeans_boundaries()

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


x,y=make_blobs(random_state=1)
kmeans=KMeans(n_clusters=3)
kmeans.fit(x)

print('클러스터 레이블:\n{}'.format(kmeans.labels_))
print('클러스터 레이블:\n{}'.format(kmeans.predict(x)))

mglearn.discrete_scatter(x[:,0],x[:,1],kmeans.labels_,markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],[0,1,2],markers='^',markeredgewidth=2)
#cluster_centers 속성에 저장된 클러스터 중심 확인하기


## 클러스터 변경 #########################################
import matplotlib.pyplot as plt
#인위적인 2차원 데이터셋
x,y=make_blobs(random_state=1)
fig,axes=plt.subplots(1,2,figsize=(10,5))
#두개의 클러스터 중심
kmeans=KMeans(n_clusters=2)
kmeans.fit(x)
assignments=kmeans.labels_
mglearn.discrete_scatter(x[:,0],x[:,1],assignments,ax=axes[0])

#다섯개 클러스터 중심
kmeans=KMeans(n_clusters=5)
kmeans.fit(x)
assignments=kmeans.labels_
mglearn.discrete_scatter(x[:,0],x[:,1],assignments,ax=axes[1])


## 클러스터의 밀도가 다를때 kmean으로 찾은 클러스터 할당 ###########
x_varied,y_varied=make_blobs(n_samples=200,cluster_std=[1.0,2.5,0.5],random_state=170)
y_pred=KMeans(n_clusters=3,random_state=0).fit_predict(x_varied)

mglearn.discrete_scatter(x_varied[:,0],x_varied[:,1],y_pred)
plt.legend(['cluster 0','cluster 1','cluster 2'],loc='best')
plt.xlabel('attr 0')
plt.ylabel('attr 1')
#--> kMean은 클러스터에 모든 방향이 똑같이 중요하다고 가정
#--> 가운데 비교적 엉성한 영역에 비해 클러스트는 중심에서 멀리 떨어진 포인트들도 포함하고있다.

## 원형이 아닌 클러스터를 구분하지 못하는 kmean ###########
import numpy as np
x,y=make_blobs(random_state=170,n_samples=600)
rng=np.random.RandomState(74)
#데이터가 길게 늘어지도록 변경
transformation=rng.normal(size=(2,2))
x=np.dot(x,transformation)

#3개의 클러스터로 데이터 kmeans 적용
kmeans=KMeans(n_clusters=3)
kmeans.fit(x)
y_pred=kmeans.predict(x)

#클러스터 할당과 클러스터 중심 나타내기
mglearn.discrete_scatter(x[:,0],x[:,1],kmeans.labels_,markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],
                         [0,1,2],markers='^',markeredgewidth=2)
plt.legend(['cluster 0','cluster 1','cluster 2'],loc='best')
plt.xlabel('attr 0')
#--> 가장 가까운 클러스터 중심까지의 거리만 고려하기 때문에 이런 데이터를 잘 처리하지 못한다.

## exam #####
#1. datasets.load_iris(), KMeans(n_clusters=3) 로 모델 생성하고 훈련시켜 예측
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
y=iris.target
kmeans=KMeans(n_clusters=3)
kmeans.fit(x)
pred=kmeans.predict(x)
kmeans.predict([[6.4,2.9,5.0,1.6]])
#2. 1의 결과를 scatter plot으로 표현
mglearn.discrete_scatter(x[:,0],x[:,1],pred,markers='o')
mglearn.discrete_scatter(x[:,2],x[:,3],pred,markers='o')

