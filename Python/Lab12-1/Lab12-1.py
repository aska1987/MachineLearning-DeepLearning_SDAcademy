# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 09:35:12 2018

@author: SDEDU
"""
'''
1. Preliminary Analysis
#Numberic variables (연속 변수)
histogram, boxplot, analysis for numeric values by non-numeric values

#Non-numeric variables
frequency table, pivot_table,crosstab

#Numeric to non-numeric variables
pd.cut(bins=break_points)

2. Missing and Extreme Values
#Missing values
df.isna().sum()
Value_counts() → fillna()
fillna(mean(), mode(), ….,inplace=True)

#Extreme values
Log transformation – np.log()

3. Encoding
#Dummy variables
pd.get_dummies()-> pd.concat() -> df.drop
#LabelEncoder in sklearn.preprocessing
le = LabelEncoder()
le.fit_transform()
'''

import numpy as np
import pandas as pd

df=pd.read_excel('encoding.xlsx')
df.shape
df.dtypes

#encoding with dummy 
a=pd.get_dummies(df)
b=pd.get_dummies(df.Country)
a=a.drop(['Country_Spain','Purchased_No'],axis=1)

#encoding with Label encoder
from sklearn.preprocessing import LabelEncoder
df=pd.read_excel('encoding.xlsx)
le=LabelEncoder()
df.Country=le.fit_transform(df.Country)
df.Purchased=le.fit_transform(df.Purchased)

#merge example
left=pd.DataFrame({'id':[1,2,3,4,5],
                   'Name':['Alex','Amy','Allen','Alice','Ayoung'],
                   'subject_id':['sub1','sub2','sub4','sub6','sub5']})
right = pd.DataFrame({'id':[1,2,3,4,5],
                      'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
                      'subject_id':['sub2','sub4','sub3','sub6','sub5']})
pd.merge(left,right,on=['id','subject_id'],how='left')
pd.merge(left,right,on=['id','subject_id'],how='right')
pd.merge(left,right,on=['id','subject_id'],how='outer')
pd.merge(left,right,on=['id','subject_id'],how='inner') #defalt
pd.merge(left,right,on=['id','subject_id'],how='right')

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                    index=[0, 1, 2, 3])
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                     index=[4, 5, 6, 7])
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                    'B': ['B8', 'B9', 'B10', 'B11'],
                    'C': ['C8', 'C9', 'C10', 'C11'],
                    'D': ['D8', 'D9', 'D10', 'D11']},
                    index=[8, 9, 10, 11])
frames = [df1, df2, df3]
result = pd.concat(frames)
result = pd.concat(frames, keys=['x', 'y', 'z'])
result.loc['y']

df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],
                    'D': ['D2', 'D3', 'D6', 'D7'],
                    'F': ['F2', 'F3', 'F6', 'F7']},
                    index=[2, 3, 6, 7])
result = pd.concat([df1, df4], axis=1, sort=False)
result = pd.concat([df1, df4], axis=1, join='inner')
result = pd.concat([df1, df4], axis=1, join_axes=[df1.index])
result = df1.append(df2)
result = df1.append(df4)
result = df1.append([df2, df3])
result = pd.concat([df1, df4], ignore_index=True)
 result = df1.append(df4, ignore_index=True)

#apply
df=pd.DataFrame([[4,9],]*3,columns=['A','B'])
df.apply(np.sqrt)
df.apply(np.sum,axis=0)
df.apply(np.sum,axis=1)
df.apply(np.log)

data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
        'year': [2012, 2012, 2013, 2014, 2014], 
        'reports': [4, 24, 31, 2, 3],
        'coverage': [25, 94, 57, 62, 70]}
df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])
df
capitalizer = lambda x: x.upper()
df['name'].apply(capitalizer)
df['name'].map(capitalizer)
#http://www.leejungmin.org/post/2018/04/21/pandas_apply_and_map/
#map 함수 = DataFram타입이 아니라 반드시 Series 타입에만 사용
#Series =값(value) + 인덱스(index)
#apply 함수 = DataFrame 에서 복수 개의 컬럼이 필요하면 사용
df = df.drop('name', axis=1)
df.applymap(np.sqrt)

a=np.arange(10,25,5)
np.linspace(0,2,9)
a[: :-1]
