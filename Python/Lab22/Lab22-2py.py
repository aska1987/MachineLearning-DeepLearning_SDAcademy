# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 15:41:04 2018

@author: SDEDU
"""
import pandas as pd
train=pd.read_csv('train1.csv')
test=pd.read_csv('test1.csv')

#basic exploration
train.shape
a=train.describe()

train.dtypes

train.Gender.value_counts()
train.Married.value_counts()
train.Dependents.value_counts()
train.Education.value_counts()
train.Self_Employed.value_counts()
train..value_counts()
train..value_counts()

#grph representation
#preprocessing



#preprocessing
#missing values
train.isnull().sum()
train.Gender.fillna(train.Gender.mode()[0],inplace=True)
train.Married.fillna(train.Married.mode()[0],inplace=True)
train.Dependents.fillna(train.Dependents.mode()[0],inplace=True)
train.Self_Employed.fillna(train.Self_Employed.mode()[0],inplace=True)
train.Credit_History.fillna(train.Credit_History.mode()[0],inplace=True)

a=train.groupby(by='Education')['LoanAmount'].median()

train.Loan_Amount_Term=train.groupby(by='Education').transform(lambda x:x.fillna(x.median()))
train.LoanAmount=train.LoanAmount.fillna(train.LoanAmount.median())
#encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
col=['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
for i in col:
    train[i]=le.fit_transform(train[i])

#logistic regression
x=train[['Credit_History','Gender','Married','Education','Loan_ID','Loan_Status']]
x=train.drop(['Loan_ID','Loan_Status'],axis=1)
y=train.Loan_Status

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x,y)
lr.predict()

log_cv=cross_val_score(lr,x,y,cv=5)
#decision tree
from sklearn.tree import DecisionTreeClassfi
#random forest
#lda
#knn
#svm

#SVM Tutorial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bank=pd.read_csv('data_banknote_authentication.txt',header=None)
bank.columns=['variance','skewness','kurtosis','entropy','class']
bank.shape

#x and y split
x=bank.drop('class',axis=1)
y=bank['class']

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)
from sklearn.svm import SVC
svc_class = SVC(kernel='linear')
svc_class.fit(x_train,y_train)
svc_class.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_train,svc_class.predict(x_train))
confusion_matrix(y_train,svc_class.predict(x_train))
confusion_matrix(y_test,svc_class.predict(x_test))
print(classification_report(y_test,svc_class.predict(x_test)))

svc_poly = SVC(kernel='poly') #곡선
svc_poly.fit(x_train,y_train)
svc_poly.predict(x_test)
accuracy_score(y_train,svc_poly.predict(x_train))
confusion_matrix(y_train,svc_poly.predict(x_train))
confusion_matrix(y_test,svc_poly.predict(x_test))
print(classification_report(y_test,svc_poly.predict(x_test)))

svc_rbf = SVC(kernel='rbf') #
svc_rbf.fit(x_train,y_train)
svc_rbf.predict(x_test)
accuracy_score(y_train,svc_rbf.predict(x_train))
confusion_matrix(y_train,svc_rbf.predict(x_train))
confusion_matrix(y_test,svc_rbf.predict(x_test))
print(classification_report(y_test,svc_rbf.predict(x_test)))

svc_sig = SVC(kernel='sigmoid') #
svc_sig.fit(x_train,y_train)
svc_sig.predict(x_test)
accuracy_score(y_train,svc_sig.predict(x_train))
confusion_matrix(y_train,svc_sig.predict(x_train))
confusion_matrix(y_test,svc_sig.predict(x_test))
print(classification_report(y_test,svc_sig.predict(x_test)))

