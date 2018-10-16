# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 09:44:35 2018

@author: SDEDU
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
foo=pd.Categorical(['a','b'],
                   categories=['a','b','c'])
bar=pd.Categorical(['d','e'],
                   categories=['d','e','f'])
foo
bar
pd.crosstab(foo,bar)

a = np.array(["foo", "foo", "foo", "foo", "bar", "bar",
              "bar", "bar", "foo", "foo", "foo"], dtype=object)
b = np.array(["one", "one", "one", "two", "one", "one",
              "one", "two", "two", "two", "one"], dtype=object)
c = np.array(["dull", "dull", "shiny", "dull", "dull", "shiny",
              "shiny", "dull", "shiny", "shiny", "shiny"],
               dtype=object)
a
b
c
pd.crosstab(a,b)
pd.crosstab(a,[b,c])
pd.crosstab(a, [b, c], rownames=['a'], colnames=['b', 'c'])

#map
x = pd.Series([1,2,3], index=['one','two', 'three'])
y = pd.Series(['foo', 'bar', 'baz'],index=[1,2,3])
z = {1: 'A', 2: 'B', 3: 'C'}
x.map(y)
x.map(z)

def myfunc(n):
    return len(n)
x=map(myfunc,('apple','banana','cherry'))
list(x)

##train1 data
df=pd.read_csv('train1.csv')
#filling missing values
a=df.describe()
a=df.isnull().sum()
a

#filling missing values
 #연속 변수
df.LoanAmount=df.LoanAmount.fillna(df.LoanAmount.mean()) 
# = df.LoanAmount.fillna(df.LoanAmount.mean(),inplace=True)
 #범주형 변수
df.Gender.value_counts()
df.Gender.fillna('Male',inplace=True)
# = df.Gender[df.Gender.isnull()==True]='Male'

#Married
df.Married.value_counts()
df.Married.fillna('Yes',inplace=True)
df.isnull().sum()
#Dependents
df.Dependents.value_counts()
df.Dependents.fillna('0',inplace=True)
df.isnull().sum()
#Self_Employed
df.Self_Employed.value_counts()
df.Self_Employed.fillna('No',inplace=True)

#LoanAmount
df.LoanAmount.fillna(df.LoanAmount.mean(),inplace=True)
df.isnull().sum()
#Loan_Amount_Term
df.Loan_Amount_Term.value_counts()
df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.mean(),inplace=True)
#Credit_History
df.Credit_History.value_counts()
df.Credit_History.mode()[0]
df.Credit_History.fillna(1.0,inplace=True)

#extreme values
df.boxplot(column='LoanAmount')
a=np.log(df.LoanAmount)
df['LoanAmount_log']=np.log(df.LoanAmount)
df.LoanAmount_log.hist(bins=50)
#ApplicantIncome
df.boxplot(column='ApplicantIncome')
df['ApplicantIncome']=np.log(df.ApplicantIncome)

#CoapplicantIncome
df.boxplot(column='CoapplicantIncome')
df['CoapplicantIncome']=np.log(df.CoapplicantIncome)
# -inf solution
df['TotalIncome']=df.ApplicantIncome+df.CoapplicantIncome
df['TotalIncome_log']=np.log(df.TotalIncome)

#DebtRatio Analysis

#encoding
#한 컬럼씩 변경
a=df.describe(include='all')
dummies=pd.get_dummies(df.Gender)
df=pd.concat([df,dummies],axis=1)
df=df.drop(['Gender','male'],axis=1)
# 0 - male, 1 - female
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df.Married=le.fit_transform(df.Married)
# 0 - No , 1 - Yes

#여러 컬럼 동시에 변경
var=['Dependents','Education','Self_Employed',
     'Property_Area','Loan_Status']
for i in var:
    df[i]=le.fit_transform(df[i])
df.dtypes
