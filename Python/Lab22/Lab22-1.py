# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:47:58 2018

@author: SDEDU
"""

import pandas as pd
tips = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')

tips.groupby(by='sex').mean()
tips.groupby(by='smoker').mean()
tips.groupby(by='day').mean()
tips.groupby(by='time').mean()
tips.groupby(by='day').transform(lambda x: x.mean())
tips.groupby(by='day')['total_bill'].transform(lambda x: x.mean())
tips['day_average']=tips.groupby(by='day')['total_bill'].transform(lambda x: x.mean())
tips.groupby(by='day')['total_bill'].transform(lambda x: x.fillna(x.median()))
tips

#bigmart example
train=pd.read_csv('Train_BigMart.csv')

test=pd.read_csv('Test_BigMart.csv')
test['Item_Outlet_Sales']=1
df=pd.concat([train,test])
df.shape
df.isnull().sum()
df.Item_Visibility.unique()
df.Item_Visibility.replace(0,df.Item_Visibility.median(),inplace=True)
df.Outlet_Size.unique()
df.Outlet_Size.fillna('Other',inplace=True)
df.Item_Fat_Content.unique()
df.Item_Fat_Content.replace(['low fat','LF'],'Low Fat',inplace=True)
df.Item_Fat_Content.replace('reg','Regilar',inplace=True)
a=df.groupby(by='Item_Type').median()
a=a.Item_Weight
df.Item_Weight.fillna(a[df.Item_Identifier])
df['Item_Weight']=df.groupby(by='Item_Identifier').transform(lambda x:x.fillna(x.median()))
a=df.groupby(by='Item_Identifier').median()
df['Year']=2018 - df.Outlet_Establishment_Year

#encoding 
#Item_Fat_Content
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df.Item_Fat_Content=le.fit_transform(df.Item_Fat_Content)
#Outlet_Size
df.Outlet_Size=le.fit_transform(df.Outlet_Size)
#Outlet_Location_Type
df.Outlet_Location_Type=le.fit_transform(df.Outlet_Location_Type)
#Outlet_Type
df.Outlet_Type=le.fit_transform(df.Outlet_Type)
#Item_Type
df.Item_Type=le.fit_transform(df.Item_Type)

#check
df.Item_Fat_Content.unique()

#drop unnecessary fields
df.drop(['Item_Identifier','Outlet_Identifier','Item_Fat_Content',
         'Outlet_Establishment_Year'],axis=1,inplace=True)

#divide it into train and test
new_train=df[:8523]
new_test=df[8523:]

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
