# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:45:23 2018

@author: SDEDU
"""

import numpy as np
import pandas as pd

df=pd.read_csv('C:\\Users\\SDEDU\\Documents\\GitHub\\MachineLearning-DeepLearning_SDAcademy\\Python\\Lab11-1\\train1.csv')

#variable selection 1
y=df.iloc[:,]
x=df.iloc[:,1:12]

#variable selection 2
df.columns
y=df[['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']]
x=df['Loan_Status']

#variable selection 3
y=df.drop(['Loan_ID','Loan_Status'],axis=1)
x=df.Loan_Status

#boston housing dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

boston=datasets.load_boston()
boston.keys()
x=boston.data
y=boston.target
boston.feature_names
print(boston.DESCR)

df=pd.DataFrame(boston.data,
               columns=boston.feature_names)

df['PRICE']=boston.target
#check the properties of the dataframe
df.head()
df.shape
df.index
df.columns
df.dtypes
df.ndim
df.size
df.values
df.info()
df.count()

#summary statistics
df_desc=df.describe()
#histogram
df.hist(bins=50)
#boxplot
df.boxplot()
#
#Correlation matrix
df.corr()
plt.matshow(df.corr())
#%matplotlib auto
pd.scatter_matrix(df)
#Scatter plot
plt.scatter(df.LSTAT,df.PRICE)
plt.scatter(df.RM,df.PRICE)
#scatter_matrix
pd.scatter_matrix(df)
#regression model ( y = ax )
import statsmodels.api as sm
model=sm.OLS(y,x).fit()
predictions=model.predict(x)
model.summary()

#regression model with y-intercept(y 절편) (y = ax + b)
x=sm.add_constant(x) #기존의 x값에다 constant 대입
model=sm.OLS(y,x).fit()
predictions=model.predict(x)
model.summary()

#regression model with RM

#regression model with y-intercept and RM
x=sm.add_constant(x)
x=sm.add_constant(x[:,6])
model=sm.OLS(y,x).fit()
predictions=model.predict(x)
model.summary()


#multi figure 
fig=plt.figure()
ax1=fig.add_subplot(2,1,1)
ax2=fig.add_subplot(2,1,2)
for i in range(1,5,1):
    plt.subplot(2,2,i)
    plt.boxplot(df.iloc[:,(i-1)])
    plt.title(df.columns[(i-1)])
#correlation matrix
a=df.corr()


#golfing example
distance = np.array([277.6,259.5,269.1,267,255.6,272.9])
score=np.array([69,71,70,70,71,69])
x=distance
y=score
x.mean()
y.mean()
cov=((x-x.mean())*(y-y.mean())).sum()/(len(x)-1)
np.std(x)
v=sum((x-np.mean(x))**2)/6
math.sqrt(v)

corr=cov/(np.std(x)*np.std(y))
corr

#numpy
a=np.vstack((x,y))
np.cov(a)
np.corrcoef(a) #strong negative relationship
#pandas
df=pd.DataFrame(a.T,columns=['distance','score'])
df.cov()
df.corr()
#scatter plot
plt.scatter(x,y)
pd.scatter_matrix(df)
