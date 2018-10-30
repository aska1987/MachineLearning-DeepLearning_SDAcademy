# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:40:16 2018

@author: SDEDU
"""

import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#load dataset
diabetes = datasets.load_diabetes()

#x and y
df=pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
y=diabetes.target

#train test split
x_train,x_test,y_train,y_test = train_test_split(df,y,test_size=.2)
x_train.shape
x_test.shape
y_train.shape
y_test.shape

#model building
lm=LinearRegression()
model=lm.fit(x_train,y_train)

#prediction
pred=model.predict(x_test)
pred

#model evaluation
r2_score(y_train,model.predict(x_train)) 
model.score(x_train,y_train)
model.score(x_test,y_test)
'''
= 0.5171.. => 낮은 예측률 
따라서 리그레이션 말고 다른 모델로 예측하기 추천
'''
plt.scatter(y_train,model.predict(x_train))
plt.xlabel('true values')
plt.ylabel('predictions')
plt.title('train')

plt.scatter(y_test,pred)
plt.xlabel('true values')
plt.ylabel('predictions')
plt.tile('test')

#cross validation
from sklearn.cross_validation import cross_val_score,cross_val_predict
scores=cross_val_score(model,x_train,y_train,cv=6)
pred=cross_val_predict(model,x_train,y_train,cv=6)
plt.scatter(y,pred)

#k fold cross validation
import numpy as np
from sklearn.model_selection import KFold
x=np.array([[1,2],[3,4],[1,2],[3,4]])
y=np.array([1,2,3,4])
kf=KFold(n_splits=2)
kf.get_n_splits(x)

for train_index, test_index in kf.split(x):
    print('train:',train_index, 'test:', test_index)
    print(y[train_index],y[test_index])

#leave one out
from sklearn.model_selection import LeaveOneOut
x=np.array([[1,2],[3,4]])
y=np.array([1,2])
loo =LeaveOneOut()
loo.get_n_splits(x)

for train_index,test_index in loo.split(x):
    print('train:',train_index,'test:',test_index)
    print(y[train_index],y[test_index])

#iris flower example
from sklearn import datasets
iris=datasets.load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
y=iris.target

#train and test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=.2,random_state=0)

#model building
#1) decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score,accuracy_score,confusion_matrix,classification_report
from sklearn.cross_validation import cross_val_score
dtree=DecisionTreeClassifier()
dtree.fit(x_train,y_train)
pred=dtree.predict(x_test) #prediction
r2_score(y_train,dtree.predict(x_train)) #evaluation
accuracy_score(y_train,dtree.predict(x_train))
dtree_cv=cross_val_score(dtree,x_train,y_train,cv=5)

confusion_matrix(y_test,dtree.predict(x_test)) #confusion matrix
confusion_matrix(y_train,dtree.predict(x_train))
print(classification_report(y_train,dtree.predict(x_train)))
#2) random forest
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
pred=rf.predict(x_test)
r2_score(y_train,rf.predict(x_train)) #evaluation
accuracy_score(y_train,rf.predict(x_train))
rf_cv=cross_val_score(rf,x_train,y_train,cv=5)

confusion_matrix(y_test,rf.predict(x_test)) #confusion matrix
confusion_matrix(y_train,rf.predict(x_train))
print(classification_report(y_train,rf.predict(x_train)))
#3) lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)
pred=lda.predict(x_test)
r2_score(y_train,lda.predict(x_train)) #evaluation
accuracy_score(y_train,lda.predict(x_train))
lda_cv=cross_val_score(lda,x_train,y_train,cv=5)

confusion_matrix(y_test,lda.predict(x_test)) #confusion matrix
confusion_matrix(y_train,lda.predict(x_train))
print(classification_report(y_train,lda.predict(x_train)))
#4) knn
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
r2_score(y_train,knn.predict(x_train)) #evaluation
accuracy_score(y_train,knn.predict(x_train))
knn_cv=cross_val_score(knn,x_train,y_train,cv=5)

confusion_matrix(y_test,knn.predict(x_test)) #confusion matrix
confusion_matrix(y_train,knn.predict(x_train))
print(classification_report(y_train,knn.predict(x_train)))
#5) svm
from sklearn.svm import SVC
svm=SVC()
svm.fit(x_train,y_train)
pred=svm.predict(x_test)
r2_score(y_train,svm.predict(x_train)) #evaluation
accuracy_score(y_train,svm.predict(x_train))
svm_cv=cross_val_score(svm,x_train,y_train,cv=5)

confusion_matrix(y_test,svm.predict(x_test)) #confusion matrix
confusion_matrix(y_train,svm.predict(x_train))
print(classification_report(y_train,svm.predict(x_train)))
#comparison among different methods
dtree_cv.mean()
rf_cv.mean()
lda_cv.mean()
knn_cv.mean()
svm_cv.mean()

dtree_cv.std()
rf_cv.std()
lda_cv.std()
knn_cv.std()
svm_cv.std()

#comparison plot
cv_summary=pd.DataFrame({'dtree':dtree_cv,
                         'rf':rf_cv,
                         'lda':lda_cv,
                         'knn':knn_cv,
                         'svm':svm_cv})
cv_summary.boxplot()
cv_summary.hist()

#pima indian diabetes example
pima=pd.read_csv('C:\\Users\\SDEDU\\Documents\\GitHub\\MachineLearning-DeepLearning_SDAcademy\\Python\\Lab20\\pima-indians-diabetes.csv',header=None)
pima.columns=['preg','plas','pres','skin','test','mass','pedi','age','class']

#train test split
x=pima.drop('class',axis='columns')
y=pima['class'].values

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=.2)
x_train.shape
y_train.shape
x_test.shape
y_test.shape
#model building
lm=LinearRegression()
model=lm.fit(x_train,y_train)

#prediction
pred=model.predict(x_test)
pred

#model evaluation
r2_score(y_train,model.predict(x_train)) 
model.score(x_train,y_train)
model.score(x_test,y_test)

#cross validation
from sklearn.cross_validation import cross_val_score,cross_val_predict
scores=cross_val_score(model,x_train,y_train,cv=10)
pred=cross_val_predict(model,x_train,y_train,cv=10)

np.mean(scores)
