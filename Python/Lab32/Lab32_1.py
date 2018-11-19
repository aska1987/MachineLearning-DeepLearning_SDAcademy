# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 14:39:27 2018
make_moons 데이터 시각화
유방암 데이터 시각화
bmi SVM분석
@author: SDEDU
"""

from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn
a=make_moons()
x,y=make_moons(n_samples=100,noise=0.25,random_state=3)
x_train,x_test,y_train,t_test=train_test_split(x,y,stratify=y,random_state=42)

forest=RandomForestClassifier(n_estimators=5,random_state=2)
forest.fit(x_train,y_train)

fig,axes=plt.subplots(2,3,figsize=(20,10))
for i,(ax,tree) in enumerate(zip(axes.ravel(),forest.estimators_)):
    print(i)
    print(ax)
    ax.set_title('tree{}'.format(i))
    mglearn.plots.plot_tree_partition(x,y,tree,ax=ax)
    
mglearn.plots.plot_2d_separator(forest,x,fill=True,ax=axes[-1,-1],alpha=.4)
axes[-1,-1].set_title('random forest')
mglearn.discrete_scatter(x[:,0],x[:,1],y)


##유방암 데이터 시각화#####################
import numpy as np
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
forest=RandomForestClassifier(n_estimators=100,random_state=0)
forest.fit(x_train,y_train)

forest.score(x_train,y_train)
forest.score(x_test,y_test)

forest.feature_importances_#특성 중요도

#특성 중요도 시각화
def plot_feature_importances_cancer(model):
    n_features=data.data.shape[1]
    plt.barh(range(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),data.feature_names)
    plt.xlabel('attr importances')
    plt.ylabel('attr')
    plt.ylim(-1,n_features)
    
plot_feature_importances_cancer(forest)


##bmi 데이터 ######################
from sklearn import svm
data=pd.read_csv('bmi.csv')
x=data[['height','weight']]
y=data['label']

from sklearn import preprocessing
encoder=preprocessing.LabelEncoder()
y=encoder.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
clf=svm.SVC(gamma='auto')
clf.fit(x_train,y_train)
clf.score(x_test,y_test)

mglearn.discrete_scatter(x['height'],x['weight'],y)
plt.legend()

x_data=pd.read_csv('bmi.csv',index_col=2)

def plot_scatter(lbl,color):
    b=x_data.loc[lbl]
    ax.scatter(b['weight'],b['height'],c=color,label=lbl)

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
plot_scatter('fat','red')
plot_scatter('normal','yellow')
plot_scatter('thin','purple')
ax.legend()

