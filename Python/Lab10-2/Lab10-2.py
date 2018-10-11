# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:58:36 2018

@author: SDEDU
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('train.csv',header=None)
loan_id=pd.read_csv('id.csv',header=None)

data=pd.concat((loan_id,df),axis=1)
var=pd.read_csv('variables.csv')
var.variable
data.columns=var.variable

#exploring data
data.head()
data.describe() #기술통계

##numeric data analysis
#boxplot using df.boxplot for ApplicantIncome
data.boxplot(column='ApplicantIncome')
#CoapplicantIncom
data.boxplot(column='CoapplicantIncome')
#Dependents
data.boxplot(column='Dependents')
#LoanAmount
data.boxplot(column='LoanAmount')
#Loan_Amount_Term
data.boxplot(column='Loan_Amount_Term')
#boxplot
data.boxplot(column=['ApplicantIncome','CoapplicantIncome','Dependents','LoanAmount',
                     'Loan_Amount_Term'])

#histogram using df.hist for ApplicantIncome
data.ApplicantIncome.hist(bins=50)
plt.xlabel('Applicant Incom(in thousands)')
plt.ylabel('Number of Applicants')
plt.title('Histogram for Applicant Income')
#Dependents
data.Dependents.hist(bins=50)
#histogram using df.hist for LoanAmount
data.LoanAmount.hist(bins=50)
#histogram using df.hist for Loan_Amount_Term
data.Loan_Amount_Term.hist(bins=50)

##non-numeric data analysis
#Gender, Married ,Education, Self_Employed,Credit_History,
#Property_Area,Loan_Status
data.Gender.value_counts()
data['Married '].value_counts()
data.Education.value_counts()
data.Self_Employed.value_counts()
data.Credit_History.value_counts()
data.Property_Area.value_counts()
data.Loan_Status.value_counts()

#pivot table 
df = pd.DataFrame({"item": ["foo", "foo", "foo", "foo", "foo",
                          "bar", "bar", "bar", "bar"],
                   "level": ["one", "one", "one", "two", "two",
                          "one", "one", "two", "two"],
                   "size": ["small", "large", "large", "small",
                          "small", "large", "small", "small",
                          "large"],
                   "number": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                   'weight':[11,12,12,31,13,14,15,16,19]})
pd.pivot_table(df,values='number',index=['item','level'],
               columns=['size'],aggfunc=np.sum)
pd.pivot_table(df,values=['number','weight'],index=['item','level'],
               columns=['size'],aggfunc={'number':np.mean,
               'weight':[min,max,np.mean]})
#Pivot table an bar plot using df.pivot_table for Loan_Status by Credit_History
temp=pd.pivot_table(data,values='Loan_Status',index=['Credit_History'],
               aggfunc=np.mean)
temp.plot(kind='bar')
plt.xlabel('Credit History')
plt.ylabel('Count of Applicants')
plt.title('Probability of Getting Loan by Credit History')
plt.show()

#Dependents, Education
data.Dependents.value_counts()
data.Education.value_counts()
Dependents=pd.pivot_table(data,values='Dependents',index=['Education'],
               aggfunc=np.mean)
Education=pd.pivot_table(data,values='Education',index=['Dependents'],
               aggfunc=np.mean)

Dependents.plot(kind='bar')
Education.plot(kind='bar')

plt.scatter(data['Dependents'],data['Education'])

