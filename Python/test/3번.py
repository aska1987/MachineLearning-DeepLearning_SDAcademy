# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:04:00 2018

@author: SDEDU
"""
import pandas as pd
import numpy as np
#1)
fpe=pd.read_csv('effort.csv')

#2)
fpe.shape

#3)
fpe.describe()
fpe.columns
fpe.head()
#4)
fpe_array=fpe.as_matrix()
#5)
fpe.describe()

#6)
fpe.effort.mean()

#7)
fpe[fpe.effort==0]

#8)
fpe[fpe.country=='Chile']

#9)
fpe.loc[4:20,['setting','effort']]
#10)
fpe.query('effort==0 & change==(1,2)')
