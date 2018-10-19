# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 09:41:17 2018

@author: SDEDU
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set(color_codes=True)
tips=sns.load_dataset('tips')
#linear regression model
sns.regplot(x='total_bill',y='tip',data=tips)
sns.lmplot(x='total_bill',y='tip',data=tips,hue='smoker')
sns.lmplot(x='total_bill',y='tip',data=tips,hue='smoker',markers=['o','x'])
sns.lmplot(x='total_bill',y='tip',data=tips,col='time')
sns.lmplot(x='total_bill',y='tip',data=tips,col='day',col_wrap=2)
sns.lmplot(x='total_bill',y='tip',data=tips,hue='smoker',col='time',row='sex')
sns.lmplot(x='size',y='tip',data=tips)
sns.lmplot(x='size',y='tip',data=tips,x_jitter=.05)
sns.lmplot(x='size',y='tip',data=tips,x_estimator=np.mean)

sns.jointplot(x='total_bill',y='tip',data=tips,kind='reg')
sns.pairplot(tips,x_vars=['total_bill','size'],y_vars=['tip'],
             aspect=.8,kind='reg')
sns.pairplot(tips, x_vars=["total_bill", "size"], y_vars=["tip"],
             hue="smoker", aspect=.8, kind="reg")

#boston housing
from sklearn import datasets
boston=datasets.load_boston()
boston.data
boston.target
df=pd.DataFrame(boston.data,columns=boston.feature_names)
df['PRICE']=boston.target
df.columns

crange=(df.CRIM.max()-df.CRIM.min())/3
break_points=np.arange(0,120,30)
df['CRIM_Bins']=pd.cut(df.CRIM,bins=break_points,include_lowest=True)
df.CRIM_Bins.value_counts()
df.hist()


#lmplot (hue, marker, col, row)
sns.lmplot(x='RM',y='PRICE',data=df,hue='CHAS')
sns.lmplot(x='CRIM',y='PRICE',data=df,hue='CHAS')
sns.lmplot(x='CRIM',y='PRICE',data=df,col='CHAS')
sns.lmplot(x='CRIM',y='PRICE',data=df,row='CHAS')
sns.lmplot(x='RM',y='PRICE',data=df,hue='RM')
#regplot
sns.regplot(x='RM',y='PRICE',data=df)

#joint plot 
sns.jointplot(x='RM',y='PRICE',data=df,kind='reg')
#pairplot
sns.pairplot(df,x_vars=['RM','LSTAT'],
             y_vars=['PRICE'],kind='reg',hue='CHAS')
sns.pairplot(df)
pd.scatter_matrix(df)



anscombe=sns.load_dataset('anscombe')
sns.lmplot(x='x',y='y',data=anscombe.query("dataset=='I'"),
           scatter_kws={'s':80})
sns.lmplot(x='x',y='y',data=anscombe.query("dataset=='II'"),
           scatter_kws={'s':80})
sns.lmplot(x='x',y='y',data=anscombe.query("dataset=='II'"),
           order=2,scatter_kws={'s':80})
sns.lmplot(x='x',y='y',data=anscombe.query("dataset=='III'"),
           scatter_kws={'s':80})

#ANOVA 
datafile='PlantGrowth.csv'
data=pd.read_csv(datafile)
data.boxplot('weight',by='group',figsize=(12,8))

ctrl = data['weight'][data.group == 'ctrl']
trt1 = data['weight'][data.group == 'trt1']
trt2 = data['weight'][data.group == 'trt2']

from scipy import stats
F,p=stats.f_oneway(ctrl,trt1,trt2)
F
p

import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data.group=le.fit_transform(data.group)

x=data.group
y=data.weight
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()
sm.stats.anova_lm(model,typ=2)

#cravens example
df=pd.read_excel('cravens.xlsx')
from scipy.stats import skew
#sales
sales=df.Sales
#mean
sales.mean()
#median
sales.median()
#mode
sales.mode()[0]
#variance
sales.var()
#standard deviation
sales.std()
#coefficion of variance
(sales.std()/sales.mean())*100
#skewness(기움)
skew(sales)

#descriptive statistices for all variables
a=df.describe()
df.hist()
df.columns

#frequency tavble
srange=(sales.max()-sales.min())/5
break_points=np.arange(1000,8000,1000)
df['Sales_Bin']=pd.cut(sales,bins=break_points,include_lowest=True)
freq=df.Sales_Bin.value_counts(sort=False)
perfreq=freq/len(df.Sales_Bin)
pd.concat([freq,perfreq],axis='columns')

df.Sales.hist()
#histogram for freq, perfreq using bar chart
plt.subplot(121)
plt.bar(np.arange(6),freq.values)
plt.title('Frequency')
plt.subplot(122)
plt.bar(np.arange(6),perfreq.values)
plt.title('Percent Frequency')

#histogram for all
len(df.columns)
plt.figure()
for i in range(len(df.columns)-1):
    plt.subplot(331+i)
    plt.hist(df[df.columns[i]])
    plt.title(df.columns[i])
    
for i in range(5):
    print(i)

#다변량 분석
pd.scatter_matrix(df)
sns.pairplot(df)
sns.jointplot(df)

#Lmplot, regplot, jointplot
df.columns
sns.lmplot('Time','Sales',df)
sns.regplot('Work','Sales',df)
sns.lmplot('Work','Sales',df,hue='Time')
sns.jointplot('Time','Sales',df,kind='reg')
sns.jointplot('Work','Rating',df,kind='reg')

#correlation matrix
a=df.corr()
df.Time
df.Poten

#regression withg all variables
y=df.Sales
x=df.drop(['Sales','Sales_Bin'],axis='columns')

import statsmodels.api as sm
x=sm.add_constant(x)
model =sm.OLS(y,x).fit()
model.summary()
#r suquared value = 0.922
#1. 예측 공식
#2. r squared values
#3. F test
#4. T test

#regressiong with three variables
x=df[['Poten','AdvExp','Share']]
y=df.Sales
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()

#regression with four variables Accounts
x=df[['Poten','AdvExp','Share','Accounts']]
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()
'''
prediction equation sales :
Sales = -1441.9323 + 0.0382*Poten + 0.1750*AdvExp + 190.1442*Share + 9.2139*Accounts
r squared : 0.90

if : Poten= 20000, AdExp = 10000, Share =8, Accounts = 100
result : 3514.6113
'''

#stepwise ==> Poten, AdvExp, Share, Accounts
#forward ==> Poten, AdvExp, Share, Accounts
#backward ==> Poten, AdvExp, Share, Accounts
x=df[['Time','Poten','AdvExp','Share','Change']]
y=df.Sales
x= sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()
'''
r squared: 0.915
'''

#rfe feature selection
x=df.drop(['Sales','Sales_Bin'],axis='columns')
y=df.Sales
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
lm=LinearRegression()
rfe=RFE(lm,5)
rfe=rfe.fit(x,y)
rfe.support_
x.columns[rfe.support_]
ranking=pd.DataFrame([rfe.ranking_,x.columns])

x=df[['Share','Change','Accounts','Work','Rating']]
y=df.Sales
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary() #R squared : 0.7

#f regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest,f_regression 
x=df.drop(['Sales','Sales_Bin'],axis='columns')
y=df.Sales
model=SelectKBest(score_func=f_regression,k=5)
results=model.fit(x,y)
results.scores_
scores=pd.DataFrame(results.scores_,index=x.columns)
results.pvalues_
scores.sort_values(by=0,ascending=True)

'''
최종적으로 r squared 가 0.915인 모델로 선정
'''
#residual plot
y_pred = model.predict(x)
residual=y-y_pred
std_residual=residual/np.std(residual)
plt.scatter(y_pred,std_residual)
plt.grid()

#remove index 9 record
df=pd.read_excel('cravens.xlsx')
df=df.drop(9)

x=df[['Time','Poten','AdvExp','Share','Change']]
y=df.Sales
x= sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()
'''
제거 후 r squared : 0.925
'''
y_pred = model.predict(x)
residual=y-y_pred
std_residual=residual/np.std(residual)
plt.scatter(y_pred,std_residual)
plt.grid()
