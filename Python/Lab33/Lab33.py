# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 09:56:03 2018
MinMaxScaler로 데이터 전처리 성능 측정
@author: SDEDU
"""
'''
standardScaler
각 특성의 평균을 0 분산을 1로 변경하여 모든 특성이 같은 크기를 가지게한다
RobustScaler
특성들이 같은 스케일을 같게된다는 통계적측명에서 
MinmaxScaler
모든 값을 0-1사이로 
Normalizer

'''
## MinMaxScaler로 데이터 전처리 성능 측정
import mglearn
mglearn.plots.plot_scaling()

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer=load_breast_cancer()

x_train,x_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=1)
x_train.shape
x_test.shape

#MinMaxScaler 이용하여 전처리 후 속성값확인
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)

#fit 메서드로 학습한 변환을 적용하려면 스케일객체의 transform 메서드를 사용 -> 데이터 변환
x_train_scaled=scaler.transform(x_train)
#스케일이 조정된 후 데이터셋의 속성을 출력
print('변환된 후 크기: {}'.format(x_train_scaled.shape))
# ---->데이터 스케일을 변환해도 개수에는 변화는 없다.

print('스케일 조정 전 특성별 최소값:{}'.format(x_train.min(axis=0)))
print('스케일 조정 전 특성별 최대값:{}'.format(x_train.max(axis=0)))
print('스케일 조정 후 특성별 최소값:{}'.format(x_train_scaled.min(axis=0)))
print('스케일 조정 후 특성별 최대값:{}'.format(x_train_scaled.max(axis=0)))
# ----> 최소와 최대값이 0,1로 변환,데이터의 배열 크기는 원래 데이터와 동일, 각 특성의 값이 0-1 로 변형

x_test_scaled=scaler.transform(x_test)
print(x_test.min(axis=0))
print(x_test.max(axis=0))
print(x_test_scaled.min(axis=0))
print(x_test_scaled.max(axis=0))


##데이터 전처리 성능측정 MinMaxScaler, SVM #################
from sklearn.svm import SVC
svm=SVC(C=100)
svm.fit(x_train,y_train)
print('훈련세트정확도:{:.2f}'.format(svm.score(x_train,y_train)))
print('테스트세트정확도:{:.2f}'.format(svm.score(x_test,y_test)))
#훈련 정확도: 1.0, 테스트 정확도: 0.62

minmax_scaler=MinMaxScaler()
minmax_scaler.fit(x_train)
x_train_scaled=minmax_scaler.transform(x_train)
x_test_scaled=minmax_scaler.transform(x_test)

#조정된 데이터로 SVM 학습
svm.fit(x_train_scaled,y_train)
print('스케일 조정된 훈련 세트 정확도 :{:.2f}'.format(svm.score(x_train_scaled,y_train)))
print('스케일 조정된 테스트 세트 정확도 :{:.2f}'.format(svm.score(x_test_scaled,y_test)))
#조정 후 훈련 세트 정확도: 0.99 조정 후 테스트 세트 정확도: 0.97


## 데이터 전처리 성능 측정 StandardScaler 사용  #################
from sklearn.preprocessing import StandardScaler
svm=SVC(C=100)
standard_scaler=StandardScaler()
standard_scaler.fit(x_train)
x_train_scaled_standard=standard_scaler.transform(x_train)
x_test_scaled_standard=standard_scaler.transform(x_test)

#조정된 데이터로 SVM 학습
svm.fit(x_train_scaled_standard,y_train)
print('스케일 조정된 훈련 세트 정확도:{:.2f}'.format(svm.score(x_train_scaled_standard,y_train)))
print('스케일 조정된 테스트 세트 정확도:{:.2f}'.format(svm.score(x_test_scaled_standard,y_test)))
#조정 후 훈련 세트 정확도: 1.0, 조정 후 테스트 세트 정확도: 0.97

'''
PCA(principal component analysis) : 주성분 분석
고차원의 데이터를 저차원의 데이터로 환원시키는 기법


'''
mglearn.plots.plot_pca_illustration()

## PCA를 적용해 유방암 데이터셋 시각화
import matplotlib.pyplot as plt
import numpy as np
cancer=load_breast_cancer()
fig,axes=plt.subplots(5,6,figsize=(10,20))
malignant=cancer.data[cancer.target==0]
benign=cancer.data[cancer.target==1]
ax=axes.ravel()
for i in range(30):
    _,bins=np.histogram(cancer.data[:,i],bins=50)
    ax[i].hist(malignant[:,i],bins=bins,color=mglearn.cm3(0),alpha=.5)
    ax[i].hist(benign[:,i],bins=bins,color=mglearn.cm3(2),alpha=.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel('attr size')
ax[0].set_ylabel('frequency')
ax[0].legend(['neg','pos'],loc='best')
fig.tight_layout()

## 처음 두개의 주성분을 사용해 그린 유방암 데이터셋의 2차원 산점도 #####
cancer=load_breast_cancer()
standard_scaler=StandardScaler()    
standard_scaler.fit(cancer.data)
x_scaled=standard_scaler.transform(cancer.data)
'''
PCA객체를 생성 -> fit 메서드 호출-> 주성분을 찾고,
transform 메서를 호출 -> 데이터를 회전 차원을 축소
기본값일때 PCA는 데이터를 회전만 시키고 모든 주성분을 유지
데이터의 차원을 줄이려면 PCA객체를 지정하면 된다.
'''
from sklearn.decomposition import PCA
pca=PCA(n_components=2) #데이터 첫2개의 성분만 유지한다

#PCA 모델 만들기
pca.fit(x_scaled)
#처음 두개의 주성분을 사용해 데이터 변환
x_pca=pca.transform(x_scaled)
print('원본 데이터 형태:{}'.format(x_scaled.shape))
print('축소된 데이터 형태:{}'.format(x_pca.shape))
#원본 데이터 형태 :(569,30), 축소된 데이터 형태 :(569,2)


import pandas as pd
df=pd.DataFrame(columns=['calory','breakfast','lunch','dinner','exercise','body_shape'])
df.loc[0]=[1200,1,0,0,2,'Skinny']
df.loc[1]=[2800,1,1,1,1,'Normal']
df.loc[2]=[3500,2,2,1,0,'Fat']
df.loc[3]=[1400,0,1,0,3,'Skinny']
df.loc[4]=[5000,2,2,2,0,'Fat']
df.loc[5]=[1300,0,0,1,2,'Skinny']
df.loc[6]=[3000,1,0,1,1,'Normal']
df.loc[7]=[4000,2,2,2,0,'Fat']
df.loc[8]=[2600,0,2,0,0,'Normal']
df.loc[9]=[3000,1,2,1,1,'Fat']

x=df[['calory','breakfast','lunch','dinner','exercise']]
y=df['body_shape']

from sklearn.preprocessing import StandardScaler
x_std=StandardScaler().fit_transform(x)
x_std

#Covariance Matrix of features
import numpy as np
features=x_std.T
covariance_matrix=np.cov(features)
covariance_matrix

#eigen vectors, eigen values
eig_vals,eig_vecs=np.linalg.eig(covariance_matrix)
eig_vecs

eig_vals[0]/sum(eig_vals)

#Project data point onto selected eigen vector
projected_x=x_std.dot(eig_vecs.T[0])
projected_x

result=pd.DataFrame(projected_x,columns=['PC1'])
result['y-axis']=0.0
result['label']='Y'

import seaborn as sns
sns.lmplot('PC1','y-axis',data=result,fit_reg=False,
           scatter_kws={'s':50},hue='label')
plt.title('PCA result')

from sklearn import decomposition
pca=decomposition.PCA(n_components=1)
sklearn_pca_x=pca.fit_transform(x_std)

sklearn_result=pd.DataFrame(sklearn_pca_x,columns=['PC1'])
sklearn_result['y-axis']=0.0
sklearn_result['label']='Y'
sns.lmplot('PC1','y-axis',data=sklearn_result,fit_reg=False,
           scatter_kws={'s':50},hue='label')


## 유방암 데이터셋에서 찾은 처음 두개의 주성분 히트맵 ############
import matplotlib.pyplot as plt
import sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#각 특성의 분산이 1이되도록 스케일 조정
cancer=load_breast_cancer()
standard_scaler=StandardScaler()

x_scaled=standard_scaler.transform(cancer.data)

#데이터 첫2개의 성분만 유지
pca=PCA(n_components=2)

#pca 모델 만들기
pca.fit(x_scaled)

#히트맵 시각화
plt.matshow(pca.components_,cmap='viridis')
plt.yticks([0,1],['comp 1','comp 2'])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),
           cancer.feature_names,rotation=60,ha='left')
plt.xlabel('attr')
plt.ylabel('principle comp')
plt.show()

## iris 데이터에 대해 전처리 과정을 거치고 PCA알고리즘으로 분석하기 ##
from sklearn.datasets import load_iris
iris=load_iris()
standard_scaler=StandardScaler()
standard_scaler.fit(iris.data)
x_scaled=standard_scaler.transform(iris.data)
pca=PCA(n_components=2)
pca.fit(x_scaled)
x_pca=pca.transform(x_scaled)
x_pca
