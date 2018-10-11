# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 15:17:39 2018

@author: SDEDU
"""

import pandas as pd

#Frequency
s= pd.Series([1,2,2,3,3,3,'fred',1.8,1.8])
s

s.value_counts() 
c=s.value_counts() 
c=s.value_counts() 
len(s.value_counts())  #number of unique values
s.nunique() #number of unique values
sum(s.value_counts())
c['fred'] #빈도 수
c[2]
dict(c)

s= pd.Series(['1','2','2','$3','$3','$3','fred','1.8','1.8'])
s
s.str.replace('$','') #removing

#Import the dataset and assign it to a variable
#called chipo
data=pd.read_csv('chipotle.txt',delimiter='\t')
#Display the first 10 rows
data.head
#What is the number of observations in the dataset?
data.shape[0]
#What is the number of columns in the dataset?
data.shape[1]
#Print the name of all the columns.
data.columns
#How is the dataset indexed?
data.index
#Which was the most-ordered item?
 #data.groupby(['item_name']).nunique()
data.loc[0,['quantity']]==1

most_ordered=data['item_name'].value_counts()
data.groupby(['item_name']).sum()
temp=data.groupby(['item_name']).sum().sort_values(by='quantity',ascending=False)

temp
most_ordered
#For the most-ordered item, how many items were ordered?
temp.sort_values(by='item_price',ascending=False)
data['item_name'].value_counts()
#What was the most ordered item in the choice_description column?
data['choice_description'].value_counts()
data.groupby(['choice_description']).sum().sort_values(by='quantity',ascending=False).head(1)

#How many items were orderd in total?
sum(data['quantity'].value_counts())
data.quantity.sum()
#Convert the item price into a float
data['item_price']=data['item_price'].str.replace("$","")
data['item_price'].dtypes
data['item_price']=data['item_price'].astype('float64')
data['item_price'].dtypes
#How much was the revenue for the period in the dataset?
data['item_price'].sum()
#Average of unit price(=item_price/quantity)
(data.item_price/data.quantity).mean()
#How many different items are sold?
data.item_name.nunique()
data.item_name.value_counts().count()

#missing values
import numpy as np
a=np.array([1,np.nan,3,4])
a.dtype
#detecting null values
data=pd.Series([1,np.nan,3,4,5])
data.isnull()
data.notnull()
data[data.notnull()]

#dropping null values
df=pd.DataFrame([[1,np.nan,2],[2,3,5],[np.nan,4,6]])
df
df.dropna()
df.dropna(axis=1)
df.dropna(how='all')
df.dropna(how='any')
df.dropna(thresh=3)

df.fillna(15)
df.columns=['number','numberagain','string']
df
df.fillna({'number':0,
            'numberagain':4,
            'string':3})
df.fillna(method='ffill')
df.fillna(method='bfill')
df.interpolate()
#replacing values
df=pd.DataFrame([[1,-9999,'A'],
                 [2,3,'B'],
                 [-9999,4,-9999]])
df.replace(-9999,np.nan)

df=pd.DataFrame([[1,-9999,'John'],
                 [2,25,'B'],
                 [-9999,4,-9999]])
df.columns=['a','b','c']
df
#multiple values
df.replace({25:26,
            'John':'Johnny'})
#replacing multiple columns
df.replace({'a':-9999,},0)
df.replace({'b':-9999,},np.nan)
df.replace({'c':-9999,},'None')

df.replace('[A-Za-z]','hi',regex=True)

df.replace({'a':'[A-Za-z]',
            'c':'[A-Za-z]'},"R",regex=True)

data={'name':['kim','park','lee','moon','choi'],
      'score':['good','good','bad','bad','average']}
df=pd.DataFrame(data)
df.replace(['good','bad','average'],[1,2,3])
