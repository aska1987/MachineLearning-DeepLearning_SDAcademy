# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 18:18:28 2018

@author: SDEDU
"""

#regressoion model using Loan prediction problem
loan=pd.read_csv('C:\\Users\\SDEDU\\Documents\\GitHub\\MachineLearning-DeepLearning_SDAcademy\\Python\\Lab10-2\\train.csv',header=None)
var=pd.read_csv('C:\\Users\\SDEDU\\Documents\\GitHub\\MachineLearning-DeepLearning_SDAcademy\\Python\\Lab10-2\\variables.csv')
loan.columns=var.variable.drop(0)
x=loan.drop(['Loan_Status'],axis='columns')
y=loan.Loan_Status
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()
#r squared : 0.215

#logistic regression
x=sm.add_constant(x)
model=sm.Logit(y,x).fit()
model.summary()
#r squared : 0.1705

#Logistic regression with selected variables
loan.columns
x=loan[['Married ','Credit_History','Property_Area']]
y=loan.Loan_Status
x=sm.add_constant(x)
model=sm.Logit(y,x).fit()
model.summary()
#r squ : 0.1656

#Logistic regression with Credit_History, 
#Education, Married ,Self_Employed, Property_Area
x=loan[['Credit_History','Education','Married ','Self_Employed','Property_Area']]
y=loan.Loan_Status
model=sm.Logit(y,x).fit()
model.summary()
#r squ : 0.1251
'''
그나마 제일 나은 모델: 첫번째 모델 (r squ : 0.215)
'''
