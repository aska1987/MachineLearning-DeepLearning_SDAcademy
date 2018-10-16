# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:38:56 2018

@author: SDEDU
"""

##2번
#1)
import pandas as pd
import numpy as np
df=pd.read_csv('train1.csv')

#2)
df=df.head(21)
df=df.loc[:,['Property_Area','ApplicantIncome','Loan_Status']]
#3)
df_create=pd.DataFrame([['Rural',1000],
                       ['Semiurban',5000],
                       ['Urban',12000]],columns=['Property_Area','rates'])

#4)
result = pd.merge(df_create,df,on=['Property_Area'], how='inner')

#5)
result=result.sort_values(by=['ApplicantIncome','rates'],ascending=False)

