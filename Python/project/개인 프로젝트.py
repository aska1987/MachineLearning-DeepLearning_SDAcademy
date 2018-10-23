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
import seaborn as sns
data=pd.read_csv('C:\\Users\\SDEDU\\Downloads\\london-crime\\london_crime_by_lsoa.csv',nrows=5000 )
data=pd.read_csv('crime.csv',encoding='CP949',nrows=5000,
                 usecols=['OFFENSE_CODE_GROUP','DISTRICT','MONTH','DAY_OF_WEEK','HOUR','STREET'])
data=pd.read_csv('crime.csv',encoding='CP949',
                 usecols=['OFFENSE_CODE_GROUP','DISTRICT','MONTH','DAY_OF_WEEK','HOUR','STREET'])

or_data=pd.read_csv('BreastCancerWisconsin.csv')
data=pd.read_csv('BreastCancerWisconsin.csv',usecols=[1,2,3,4,5,6,7,8,9])


'''
diagnosis : M=악성(암) , B= 양성종양 (설명 :https://m.blog.naver.com/PostView.nhn?blogId=worldphc&logNo=100187176424&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F)
radius_mean : 둘레 중앙에서 점까지의 거리의 평균
texture_mean : gray-scale values 의 표준편차
perimeter_mean : 중심 종양의 평균 크기
area_mean : 면적 평균
smoothness_mean 반지름 길이의 변화의 평균
compactness_mean : 둘레평균^2 / area -1.0
concavity_mean : 윤곽의 오목한 부분의 심각도의 평균
concave points_mean : 형상의 오목한 부분의 수를 의미
'''
data.columns
data.hist(bins=50,rwidth=0.9)


plt.hist(data['radius_mean'],histtype='bar',rwidth=0.9)
plt.xlabel('radius_mean')
plt.ylabel('frequency')
plt.title('Histogram of radius_mean')

plt.hist(data['texture_mean'],histtype='bar',rwidth=0.9)
plt.xlabel('texture_mean')
plt.ylabel('frequency')
plt.title('Histogram of texture_mean')

plt.hist(data['perimeter_mean'],histtype='bar',rwidth=0.9)
plt.xlabel('perimeter_mean')
plt.ylabel('frequency')
plt.title('Histogram of perimeter_mean')

plt.hist(data['area_mean'],histtype='bar',rwidth=0.9)
plt.xlabel('area_mean')
plt.ylabel('frequency')
plt.title('Histogram of area_mean')

plt.hist(data['smoothness_mean'],histtype='bar',rwidth=0.9)
plt.xlabel('smoothness_mean')
plt.ylabel('frequency')
plt.title('Histogram of smoothness_mean')

plt.hist(data['compactness_mean'],histtype='bar',rwidth=0.9)
plt.xlabel('compactness_mean')
plt.ylabel('frequency')
plt.title('Histogram of compactness_mean')

plt.hist(data['concavity_mean'],histtype='bar',rwidth=0.9)
plt.xlabel('concavity_mean')
plt.ylabel('frequency')
plt.title('Histogram of concavity_mean')

plt.hist(data['concave points_mean'],histtype='bar',rwidth=0.9)
plt.xlabel('concave points_mean')
plt.ylabel('frequency')
plt.title('Histogram of concave points_mean')

#statistics
deta_describe=data.describe()
data.info()
data.head()

#correlation
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['diagnosis']=le.fit_transform(data['diagnosis']) #M : 1 , B : 0
or_data['diagnosis']=le.fit_transform(or_data['diagnosis']) #M : 1 , B : 0
correlation=data.corr()
c=or_data.corr()
import matplotlib.pyplot as plt
plt.matshow(or_data.corr())
plt.matshow(data.corr())

'''
radius_mean, perimeter_mean, area_mean, concave points_mean 과 강한 상관관계
'''
total_p=data.pivot_table(values=['radius_mean','texture_mean','perimeter_mean','smoothness_mean','compactness_mean'],columns='diagnosis')

plt.bar(data.diagnosis,data.radius_mean)
plt.xlabel('diagnosis')
plt.ylabel('radius_mean')
plt.title('Barplot of radius_mean')
plt.show()

plt.scatter(data.diagnosis,data.radius_mean)
plt.xlabel('diagnosis')
plt.ylabel('radius_mean')
plt.title('Scatter of radius_mean')
plt.show()

plt.scatter(data.area_mean,data.radius_mean)
plt.xlabel('area_mean')
plt.ylabel('radius_mean')
plt.title('Scatter of area_mean, radius_mean')
plt.show()

plt.scatter(data.perimeter_mean,data.radius_mean)

plt.scatter(data.smoothness_mean,data.radius_mean)
plt.xlabel('smoothness_mean')
plt.ylabel('radius_mean')
plt.title('Scatter of smoothness_mean, radius_mean')
plt.show()

plt.scatter(data.compactness_mean,data.radius_mean)
plt.scatter(data['concavity_mean'],data.radius_mean)

plt.scatter(data['concave points_mean'],data.radius_mean)
plt.xlabel('concave points_mean')
plt.ylabel('radius_mean')
plt.title('Scatter of concave points_mean, radius_mean')
plt.show()

sns.lmplot(x='area_mean',y='radius_mean',data=data,hue='diagnosis',markers=['o','x'],y_jitter=1)
sns.lmplot(x='area_mean',y='perimeter_mean',data=data,hue='diagnosis')
sns.lmplot(x='area_mean',y='radius_mean',data=data,hue='diagnosis')
sns.lmplot(x='radius_mean',y='smoothness_mean',data=data,hue='diagnosis')

sns.jointplot('area_mean','radius_mean',data,kind='reg')
sns.jointplot('area_mean','perimeter_mean',data,kind='reg')
sns.jointplot('area_mean','smoothness_mean',data,kind='reg')


#분석 모델선정
import statsmodels.api as sm
x=data[['radius_mean','perimeter_mean','area_mean']]
y=data['diagnosis']
x=sm.add_constant(x)
model=sm.Logit(y,x).fit()
model.summary() 
'''
r squ : 0.6794

'''
x=data.drop(['diagnosis'],axis='columns')
x=sm.add_constant(x)
model=sm.Logit(y,x).fit()
model.summary() #r suq : 0.8014

x=data.drop(['diagnosis','texture_mean'],axis='columns')
x=sm.add_constant(x)
model=sm.Logit(y,x).fit()
model.summary() #r suq : 0.7352

x=data.drop(['diagnosis','area_mean'],axis='columns')
x=sm.add_constant(x)
model=sm.Logit(y,x).fit()
model.summary() #r suq : 0.7944

x=data.drop(['diagnosis','area_mean','smoothness_mean'],axis='columns')
x=sm.add_constant(x)
model=sm.Logit(y,x).fit()
model.summary() #r suq : 0.7853

x=data.drop(['diagnosis','smoothness_mean'],axis='columns')
x=sm.add_constant(x)
model=sm.Logit(y,x).fit()
model.summary() #r suq : 0.7925

x=data.drop(['diagnosis','smoothness_mean','texture_mean'],axis='columns')
x=sm.add_constant(x)
model=sm.Logit(y,x).fit()
model.summary() #r suq : 0.7335

x=data.drop(['diagnosis','concavity_mean'],axis='columns')
x=sm.add_constant(x)
model=sm.Logit(y,x).fit()
model.summary() #r suq : 0.8

x=data.drop(['diagnosis','concavity_mean','smoothness_mean'],axis='columns')
x=sm.add_constant(x)
model=sm.Logit(y,x).fit()
model.summary() #r suq : 0.7924

x=data.drop(['diagnosis','compactness_mean'],axis='columns')
x=sm.add_constant(x)
model=sm.Logit(y,x).fit()
model.summary() #r suq : 0.8011

x=data.drop(['diagnosis','compactness_mean','concavity_mean'],axis='columns')
x=sm.add_constant(x)
model=sm.Logit(y,x).fit()
model.summary() #r suq : 0.7999

#residual plot
y_pred = model.predict(x)
residual=y-y_pred
std_residual=residual/np.std(residual)
plt.scatter(y_pred,std_residual)
plt.grid()



x=or_data.drop(['diagnosis'],axis='columns')
y=or_data['diagnosis']

x=sm.add_constant(x)
model=sm.Logit(y,x).fit()
model.summary() #r suq : 0.7087

#null 갯수 count
or_data.isna().sum()

#null 데이터 삭제
or_data.columns
data=data.dropna()
del or_data['Unnamed: 32']
del or_data['id']
#요수 갯수 
data.diagnosis.value_counts()


df=data
df





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




