# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 10:51:18 2018
keras , TSNE
@author: SDEDU
"""
'''
TSNE 
고차원의 데이터를 축소시켜서 시각화하는 경우
PCA의 경우 sklearn.decomposition에 있고,
TSNE의 경우는 sklearn.manifold에 있다
데이터의 분포가 다차원인 경우 차원을 축소하기 위해 사용
'''

import seaborn as sns
import pandas as pd
import numpy as np
sns.set(style='ticks',color_codes=True)
iris=sns.load_dataset('iris',engine='python')
g=sns.pairplot(iris,hue='species',palette='husl')

iris.info()
iris['species'].unique()
from sklearn.preprocessing import LabelEncoder
x=iris.iloc[:,0:4].values
y=iris.iloc[:,4].values

encoder=LabelEncoder()
y1=encoder.fit_transform(y)
Y=pd.get_dummies(y1).values
Y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.2,
                                               random_state=1)
x_train.shape
x_test.shape
y_train.shape
y_test.shape

## Classification 이미지 식별 #########################################
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
model=Sequential()
model.add(Dense(64,input_shape=(4,),activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
model.summary()
hist=model.fit(x_train,y_train,validation_data=(x_test,y_test),
               epochs=100)
hist

import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['loss','val_loss','acc','val_acc'])
plt.grid()

loss,accuracy=model.evaluate(x_test,y_test)
accuracy

import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
y_pred=model.predict(x_test)
y_test_class=np.argmax(y_test,axis=1)
y_pred_class=np.argmax(y_pred,axis=1)
classification_report(y_test_class,y_pred_class)
confusion_matrix(y_test_class,y_pred_class)

test_set=np.array([[5,2.9,1,0.2]])
iris['species'].unique()[model.predict_classes(test_set)] #predicted target name
iris.query("species=='versicolor'")

test_set=np.array([[7,3.0,5,1.4]])
iris['species'].unique()[model.predict_classes(test_set)] #predicted target name

## word2vec ##########################################################
import logging
logging.basicConfig(
        format='%(asctime)s :%(levelname)s : %(message)s',
        level=logging.INFO)

#파라미터값 지정
num_features=300 #문자 벡터 차원 수
min_word_count=40 #최소 문자 수
num_worker=4 #병렬 처리 스레드 수
context=10 #문자열 창 크기
downsampling = 1e-3 #문자 빈도 수

#초기화 및 모델학습
from gensim.models import word2vec
sentences=[['i','you','we'],['go','come','go and back']]

#모델학습
model=word2vec.Word2Vec(sentences,min_count=1)

#학습이 완료되면 필요없는 메모리를 unload
model.init_sims(replace=True)

model_name='300features_40minwords_10text'

model.save(model_name)

from sklearn.manifold import TSNE #고차원의 데이터를 축소시켜서 시각화하는 경우에 사용
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import gensim.models as g

#그래프에서 마이너스 폰트 깨지는 몬제
mpl.rcParams['axes.unicode_minus']=False

model_name='300features_40minwords_10text'
model=g.Doc2Vec.load(model_name)

vocab=list(model.wv.vocab)
x=model[vocab]

len(x)
x[0][:10]
tsne=TSNE(n_components=2)

#100개의 단어에 대해서만 시각화
x_tsne=tsne.fit_transform(x[:100,:])

df=pd.DataFrame(x_tsne,index=vocab[:100],columns=['x','y'])
df.shape

df

model.wv.most_similar('i')

fig=plt.figure()
fig.set_size_inches(10,10)
ax=fig.add_subplot(1,1,1)

ax.scatter(df['x'],df['y'])
for word,pos in df.iterrows():
    ax.annotate(word,pos,fontsize=20)
  
    
    
    
def ngram(s,num):
    res=[]
    slen=len(s)-num+1
    for i in range(slen):
        ss=s[i:i+num]
        res.append(ss)
    return res

def diff_ngram(sa,sb,num):
    a=ngram(sa,num)
    b=ngram(sb,num)
    r=[]
    cnt=0
    for i in a:
        for j in b:
            if i ==j:
                cnt +=1
                r.append(i)
    return cnt/len(a),r

a='미드마이가 페이커를 한다.'
b='페이커가 미드마이를 한다.'

r2,word2=diff_ngram(a,b,2)
r3,word3=diff_ngram(a,b,3)
print('2-gram:',r2,word2)
print('3-gram:',r3,word3)

