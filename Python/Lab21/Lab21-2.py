# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:18:40 2018

@author: SDEDU
"""
'''
Import the train data of Big Mart Sales, do
some basic exploration tasks, and use the
graphs to visualize the data.

1)Read csv
2)shape, df.isnull().sum(), describe()
3)Distribution and frequency of sales with item visibility,
  sales by outlet identifier, sales by item type, 
  outliers and mean deviation of price by item type
'''

#BigMart data
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
#1)
train=pd.read_csv('Train_BigMart.csv')
#2)
train.shape
train.isnull().sum()
temp=train.describe()

#3)
#visualization
train.columns
plt.scatter(train.Item_Visibility,train.Item_Outlet_Sales,marker='s',c='navy')
plt.xlabel('Item Visibility')
plt.ylabel('Item Outlet Sales')
plt.title('Item Visibility vs Item Outlet Sales')


plt.bar(train.Outlet_Identifier,train.Item_Outlet_Sales)
plt.xlabel('Outlet Identifier')
plt.ylabel('item Outlet Sales')
plt.title('Outlets vs Total Sales')

bardata=train.groupby(by='Outlet_Identifier').sum()
plt.bar(bardata.index,bardata.Item_Outlet_Sales)


plt.bar(train.Item_Type,train.Item_Outlet_Sales)
plt.xticks(rotation=60)
plt.xlabel('Item Type')
plt.ylabel('Item Outlet Sales')
plt.title('Item Type vs Sales')
plt.tight_layout()

bardata=train.groupby('Item_Type').sum()
plt.bar(bardata.index,bardata.Item_Outlet_Sales)
plt.xticks(rotation=60)
plt.tight_layout()


train.dtypes
boxdata=train[['Item_Type','Item_MRP']]
boxdata.boxplot(by='Item_Type')
plt.xticks(rotation=60,color='red')


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
