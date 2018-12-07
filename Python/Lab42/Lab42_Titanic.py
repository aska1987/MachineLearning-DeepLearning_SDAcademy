# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 17:45:35 2018

@author: SDEDU
"""

import numpy as np
import pandas as pd
import tensorflow as tf
train_data=pd.read_csv('Titanic/train.csv')
test_data=pd.read_csv('Titanic/test.csv')
test_sub=pd.read_csv('Titanic/gender_submission.csv')

train_data=train_data.as_matrix()
test_data=test_data.as_matrix()
test_sub=test_sub.as_matrix()

#데이터 전처리
#male -> 1
#female -> 0
for i in range(len(train_data)):
    if train_data[i,4]=='male':
        train_data[i,4] =1
    else:
        train_data[i,4]=0
        
train_data[:,4]

for i in range(len(test_data)):
    if test_data[i,3] =='male':
        test_data[i,3]=1
    else:
        test_data[i,3]=0

#승선항 전처리 S->1, C->2, Q->3
for i in range(len(train_data)):
    if train_data[i,11] =='S':
        train_data[i,11]=1
    elif train_data[i,11] =='C':
        train_data[i,11]=2
    elif train_data[i,11] =='Q':
        train_data[i,11]=3
    if np.isnan(train_data[i,11]):
        train_data[i,11]=0
        
        
for i in range(len(test_data)):
    if test_data[i,10] =='S':
        test_data[i,10]=1
    elif test_data[i,10] =='C':
        test_data[i,10]=2
    elif test_data[i,10] =='Q':
        test_data[i,10]=3
    if np.isnan(test_data[i,10]):
        test_data[i,10]=0
        
        
x_passengerData=train_data[:,[2, #Pclass
                              4, #Sex
                              6, #SibSp
                              7, #Parch
                              11 #Embarked
                              ]]
y_survived=train_data[:,1:2] #Survived

        
Test_x_passengerData=test_data[:,[1, #Pclass
                              3, #Sex
                              5, #SibSp
                              6, #Parch
                              10 #Embarked
                              ]]
Test_y_survived=test_sub[:,1:2] #Survived



        