# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 09:51:31 2018

@author: SDEDU
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data={'Service Call':[1,2,3,4,5,6,7,8,9,10],
       'Months Since Last Service':[2,6,8,3,2,7,9,8,4,6],
       'Type of Repair':['electrical','mechanical','electrical','mechanical','electrical','electrical','mechanical','mechanical','electrical','electrical'],
       'Repair Time in Hours':[2.9,3.0,4.8,1.8,2.9,4.9,4.2,4.8,4.4,4.5]}
df=pd.DataFrame(data)

#encoding
#방법 1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Type of Repair']=le.fit_transform(df['Type of Repair'])
df['Type of Repair'].replace({0:1,1:0},inplace=True)
 
#방법 2
dummies=pd.get_dummies(df['Type of Repair'])
df=pd.concat([df,dummies],axis='columns')
df=df.drop(columns=['Type of Repair','mechanical'])
df=df.rename(columns={'electrical':'Type of Repair'})

df.columns
x=df[['Months Since Last Service','Type of Repair']]
y=df['Repair Time in Hours']

import statsmodels.api as sm
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()
'''
-prediction equation
Service time= 0.9305 + 0.3876*month + 1.2627*Type of Repair
R-squared : 85.9%
'''
#regression model for:
#types --> time
x=df['Type of Repair']
y=df['Repair Time in Hours']
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()
#month --> time
x=df['Months Since Last Service']
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary() 

#boston housing data
from sklearn.datasets import load_boston 
boston=load_boston()
boston.feature_names
x=boston.data
y=boston.target
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()

#p-value of t-test > 0.05     ==> 95% 검증
df=pd.concat([pd.DataFrame(boston.data,
                           columns=boston.feature_names),
                pd.DataFrame(boston.target, columns=['target'])],
                axis=1)
x=df.iloc[:,:-1]
y=df.target
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()
# -> r squared value= 0.741


x=df.drop(['INDUS','AGE'],axis='columns')
y=df.target
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()
# -> r squared value= 1

#regression model

