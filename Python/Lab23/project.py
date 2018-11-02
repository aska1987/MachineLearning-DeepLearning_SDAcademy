# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 17:34:09 2018

@author: SDEDU
"""
import pandas as pd
import numpy as np
import matplot
olympic=pd.read_csv('athlete_events.csv')
olympic.shape
df=olympic[olympic.Sport=='Athletics']
df

df.shape
df.isnull().sum()
df.hist()

df['Medal']=df['Medal'].fillna(0)
df=df.dropna(how='any')

df.Team.unique().size()

#encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
col=['Sex','Team']
for i in col:
    df[i]=le.fit_transform(df[i])

df['Medal'].replace(['Gold','Silver','Bronze'],1,inplace=True)

x=df[['Sex','Age','Height','Weight','Team']]
y=df['Medal']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,
                                                 random_state=1102)

#naive boyesion model
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix
gnb=GaussianNB()
gnb.fit(x_train,y_train)
gnb.predict(x_test)
y_pred=gnb.predict(x_test)
accuracy_score(y_test,y_pred) 
accuracy_score(y_train,gnb.predict(x_train))

confusion_matrix(y_test,y_pred)
#prediction
gnb.predict([[0,30,177,74,8]])







#regression model
y=new_train.Item_Outlet_Sales
x=new_train.drop('Item_Outlet_Sales',axis=1)
y_test=new_test.Item_Outlet_Sales
x_test=new_test.drop('Item_Outlet_Sales',axis=1)
import statsmodels.api as sm
x=sm.add_constant(x)
x_test=sm.add_constant(x_test)
model=sm.OLS(y,x).fit()
model.summary()


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,accuracy_score,confusion_matrix,classification_report
from sklearn.cross_validation import cross_val_score

lm=LinearRegression()
lm.fit(x,y)
lm.predict(x_test)
lm.score(x,y)
r2_score(y,lm.predict(x))
lm_cv=cross_val_score(lm,x,y, cv=5)

#decision tree
from sklearn.tree import DecisionTreeRegressor
dtree=DecisionTreeRegressor()
dtree.fit(x,y)
pred=dtree.predict(x_test) #prediction
r2_score(y,dtree.predict(x)) #evaluation
dtree_cv=cross_val_score(dtree,x,y,cv=5)

#random forest
from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor()
rf.fit(x,y)
rf.predict(x_test)
r2_score(y,rf.predict(x))
rf_cv=cross_val_score(rf,x,y,cv=5)

#comparison
reg_score, lm_cv, dtree_cv, rf_cv
lm_cv.mean()
dtree_cv.mean()
rf_cv.mean()

plt.boxplot([lm_cv,dtree_cv,rf_cv])
