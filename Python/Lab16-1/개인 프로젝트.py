# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:35:37 2018

@author: SDEDU
"""

#https://www.kaggle.com/jboysen/london-crime
#https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('C:\\Users\\SDEDU\\Downloads\\london-crime\\london_crime_by_lsoa.csv',nrows=5000 )
data=pd.read_csv('crime.csv',encoding='CP949',nrows=5000,
                 usecols=['OFFENSE_CODE_GROUP','DISTRICT','MONTH','DAY_OF_WEEK','HOUR','STREET'])
data=pd.read_csv('crime.csv',encoding='CP949',
                 usecols=['OFFENSE_CODE_GROUP','DISTRICT','MONTH','DAY_OF_WEEK','HOUR','STREET'])

data=pd.read_csv('BreastCancerWisconsin.csv',usecols=[1,2,3,4,5,6,7])

#diagnosis : M=악성 , B= 양성
#radius_mean : 둘레 중앙에서 점까지의 거리의 평균
#texture_mean : gray-scale values 의 표준편차
#perimeter_mean : 중심 종양의 평균 크기
#area_mean 
#smoothness_mean 반지름 길이의 변화의 평균?
#compactness_mean : 둘레평균^2 / area-1.0

#concavity_mean : 윤곽의 오목한 부분의 심각도의 평균
#concave points_mean : ?
#symmetry_mean : 대칭?
#fractal_dimension_mean :
#radius_se : 경계에서 중심점까지의 평균 거리에 대한 표준 오차

data.columns
data.describe()
data.info()
data.describe
data.head()

#null 갯수 count
data.isna().sum()

#null 데이터 삭제
data=data.dropna()
#요수 갯수 
data.diagnosis.value_counts()

total_p=data.pivot_table(values=['radius_mean','texture_mean','perimeter_mean','smoothness_mean','compactness_mean'],columns='diagnosis')




#범주형 변수 인코딩
#요일
np.unique(data['DAY_OF_WEEK'])
day_dict={'Sunday':0,'Monday': 1,'Tuesday':2,'Wednesday':3,'Thursday':4,
          'Friday':5,'Saturday':6}
data['DAY_OF_WEEK']=data['DAY_OF_WEEK'].map(day_dict)


total_p=data.pivot_table(values='OFFENSE_CODE_GROUP',columns='DAY_OF_WEEK',index='DISTRICT',aggfunc=sum)
total_p.head()





from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['DAY_OF_WEEK']=le.fit_transform(data['DAY_OF_WEEK'])
#

import seaborn as sns
data2=data
data2['OFFENSE_CODE_GROUP']=le.fit_transform(data2['OFFENSE_CODE_GROUP'])
sns.lmplot(x='DAY_OF_WEEK',y='OFFENSE_CODE_GROUP',data=data2)

x=data['DAY_OF_WEEK']
y=data['OFFENSE_CODE_GROUP']
x=sm.add_constant(x)
model=sm.Logit(y,x).fit()
model=sm.OLS(y,x).fit()
model.summary()




