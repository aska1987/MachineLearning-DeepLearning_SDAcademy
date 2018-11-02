# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 09:43:20 2018

@author: SDEDU
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('breast-cancer-wisconsin.data',header=None)
df.columns=['id','thickness','cellsize','cellshape','adhesion','singlecellsize',
            'nuclei','chromatin','normalnucleoli','mitoses','class']

#exploration
df.drop('id',axis=1,inplace=True)
df.shape
df.isnull().sum()

df.hist()
df.boxplot()

#x and y split
x=df.drop('class',axis=1)
y=df['class']


#train and test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)

#feature scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)

#knn model building
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)

df.replace('?',-99999,inplace=True) #다시 x와y 분리부터

#prediction
y_pred=knn.predict(x_test)

#evaluation
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred) #test set evaluation
accuracy_score(y_train,knn.predict(x_train)) #

#actual prediction
new = [[4,2,1,1,1,2,3,2,1]]
knn.predict(new)


#iris example
from sklearn import datasets
iris=datasets.load_iris()

df=pd.DataFrame(iris.data)
df.columns=['sepal_length','sepal_width','petal_length','petal_width']
df['species']=iris.target

#exploration
plt.scatter(df.sepal_length,df.sepal_width,c=df.species)

#x and y split
x=df.drop('species',axis=1)
y=df.species

#train and test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33,random_state=42)

#feature scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)

#knn model building
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)

#evaluation
from sklearn.cross_validation import cross_val_score
accuracy_score(y_test,y_pred)
cross_val_score(knn,x_train,y_train,cv=10)

#diffrent k values
def find_k(x_train,y_train,x_test): 
    k=range(1,51)
    score=[]
    for i in k:
        knn=KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train,y_train)
        y_pred=knn.predict(x_test)
        score.append(accuracy_score(y_test,y_pred))
    
    error=[]
    for i in range(len(score)):
        error.append(1-score[i])
    plt.plot(k,error)    
    error.index(min(error))

#final model
#knn model building
knn=KNeighborsClassifier(n_neighbors=20)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)

#evaluation
from sklearn.cross_validation import cross_val_score
accuracy_score(y_test,y_pred)
cross_val_score(knn,x_train,y_train,cv=10)

#finding optimal k value
find_k(x_train,y_train,x_test)
