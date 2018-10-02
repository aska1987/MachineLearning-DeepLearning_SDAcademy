# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 15:21:09 2018

@author: SDEDU
"""

import pandas as pd
import numpy as np
s=pd.Series([3,-5,7,4])
s
s=pd.Series([3,-5,7,4],index=['a','b','c','d'])
s
#data frame
data={
      'Country': ['Belgium','India','Brazil'],
      'Capital': ['Brussels','New Delhi','Brasilia'],
      'Population':[11190846,1303171035,207847528]}
df=pd.DataFrame(data,columns=['Country','Capital','Population'])
df

#panel
data={
      'Item1' :pd.DataFrame(np.random.randn(4,3))}
df
p=pd.Panel(data)

#1. data frame , dictionary
#id,name,score,subjectd
data={
     'Id':['user123','python','bigdata'],
     'Name':['lee','kim','park'],
     'Score':[60,45,50],
     'Subject':['math','computer','english']}
pd.DataFrame(data,columns=['Id','Name','Score','Subject'])

#2.data frame,2d array(row by row)
data=[[96.31,1,70],
      [96.7,1,71],
      [96.9,1,74],
      [97,1,80],
      [97.1,1,73],
      [97.1,1,75]]

pd.DataFrame(data,columns=['tf','gen','hr'])
#3. panel, 1 item 3x3 (all zeros)
#2 item 3x3 (all ones)
data={
      'Item1':pd.DataFrame(np.zeros((3,3))),
      'Item2':pd.DataFrame(np.ones((3,3)))}
p=pd.Panel(data)
#4. data frame from random number 5x6
np.random.rand(5,6)
#5. series from 7 random num between 1-100
pd.Series(np.random.randint(1,100,7))

#properties
df.shape #row,column
df.index
df.columns 
df.dtypes #
df.ndim #dimention 수
df.size #요소 수
df.values 
df.info
df.count
df.head()
df.tail()
#Statistics
df.describe() 
df.mean()
df.median()
df.sum()
df.max()
df.min()
df.var()
df.std()

#Read weather.txt file into a dataframe.
data=pd.read_csv('weather.txt')
#Print first 5 or last 3 rows of df
data.head(5)
data.tail(3)
#Get data types, index, columns, values
data.dtypes
data.index
data.columns
data.values
#Statistical summary of each column
data.describe()
#Sort records by any column (descending order)
data.sort_values(by='avg_high',ascending=[False])


#Slice the records and display the following
#columns and rows:
data.shape
#avg_low
data.loc[:,['avg_low']] 
#rows 1 to 2
data.iloc[0:2]
#avg_low and avg_high
data.loc[:,['avg_low','avg_high']]
#9 row of avg_precipitation column
data.loc[9:9,['avg_precipitation']]
#4 to 5 rows of 1 and 4 columns
data.iloc[[3,4],[0,3]]
