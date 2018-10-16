# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:07:55 2018

@author: SDEDU
"""
import numpy as np
import pandas as pd
##1번
#1)
np.random.seed(1)
df=pd.DataFrame(np.random.randn(5,3),index=['a','c','e','f','h'],
                columns=['one','two','three'])
df=df.reindex(['a','b','c','d','e','f','g','h'])

#2)
df.iloc[6:8,:]

#3)
df.isnull().sum()

#4)
df.one.fillna(df.one.mean(),inplace=True)

#5)
df.fillna(method='ffill')
#6)
df.isnull().sum()

#7)
df=df.dropna(how='any')

#8)
lamb=lambda x: np.sqrt(x)
lamb(df)

#9)
df['status']=['GOOD','BAD','GOOD','GOOD','BAD']

#10)
df.groupby('status').mean()



