# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:00:41 2018

@author: SDEDU
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df=pd.read_csv('PlayTennis.csv')

#data exploration
df.head()
df.shape
df.isnull().sum()
features=['Outlook','Temperature','Humidity','Wind','Play Tennis']
for i in features:
    print(df[i].value_counts())
    print()

import seaborn as sns
sns.set(style='darkgrid')
g=sns.FacetGrid(df)

sns.countplot(x='Outlook',data=df)

x=df.drop('Play Tennis',axis=1)
y=df['Play Tennis']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.33,
                                                 random_state=54)
#encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in range(len(features)):
    df[features[i]]=le.fit_transform(df[features[i]])

#naive boyesion model
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix
gnb=GaussianNB()
gnb.fit(x_train,y_train)
gnb.predict(x_test)
y_pred=gnb.predict(x_test)
accuracy_score(y_test,y_pred) 
accuracy_score(y_train,gnb.predict(x_train))

#prediction
gnb.predict([[0,2,0,1]])

#for loop로
models=[GaussianNB(),BernoulliNB(),MultinomialNB()]
for i in models:
    i.fit(x_train,y_train)
    y_pred=i.predict(x_test)
    print(accuracy_score(y_test,y_pred))
    print(accuracy_score(y_train,i.predict(x_train)))
    print(confusion_matrix(gnb,x,y))
    print(i.predict([[0,2,0,1]]))
    print()

df.Outlook.unique() # Sunny : 2, Overcast : 0, Rain : 1

#titanic example
titanic=pd.read_csv('titanic.csv')
titanic.head()
titanic.hist()
titanic.isnull().sum()

import seaborn as sns
sns.lmplot(x='Age',y='Fare',data=titanic,col='Sex',hue='Survived')


titanic.groupby(by='Pclass')['Survived'].mean()

titanic['AgeRange']=pd.cut(titanic.Age,[0,15,80],labels=['child','adult'])
titanic.groupby(by='AgeRange')['Survived'].mean().plot.bar()

#preprocessing
#missing values
titanic.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
titanic.isnull().sum()
titanic.dropna(how='any',axis='rows',inplace=True)

#encoding
titanic.Sex=le.fit_transform(titanic.Sex)
titanic.Embarked=le.fit_transform(titanic.Embarked)
titanic.AgeRange=le.fit_transform(titanic.AgeRange)

#x and y
x=titanic.drop(['Survived','AgeRange'],axis=1)
y=titanic.Survived

#train and test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)



