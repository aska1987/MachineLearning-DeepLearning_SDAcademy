# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 09:45:33 2018

@author: SDEDU
"""

'''
학습: 신경망에서 원하는 결과를 얻기 위해 뉴런 사이의 적당한 가중치를 알아내는 것(= 가중치를 최적화하는 것)

케라스(Keras):인공지능 코딩을 쉽게할수있는 파이썬 라이브러리

케라스 설치
(base) C:\Users\SDEDU>conda install keras

'''
import keras
#1. 인공 신경망 만들기
model=keras.models.Sequential()

#2. 멤버 함수 add()를 이용해 인공지능 계층 추가
model.add(keras.layers.Dense(1,input_shape=(1,)))

#3. 만든 모델을 어떻게 학습할지 파라미터로 지정하고 컴파일
model.compile('SGD','mse')

#4. 모델을 주어진 데이터로 학습
model.fit

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import Adam
from keras.utils import np_utils
#1. Mnist 데이터 읽어드리기
(x_train, y_train),(x_test, y_test)= mnist.load_data()

#2. 데이터를 float32 자료형으로 변환하고 정규화하기
x_train=x_train.reshape(60000,784).astype('float32')
x_test=x_test.reshape(10000,784).astype('float')
x_train /=255
x_test/=255

#3. 레이블 데이터를 0-9까지의 카테고리를 나타내는 배열로 변환하기
y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)

#4. 모델 구조 정의하기
model=Sequential()
model.add(Dense(512,input_shape=(784,))) #입력 784, 출력 512
model.add(Activation('relu'))
model.add(Dropout(0,2))
model.add(Dense(512)) #입력 512(이전 레이어 입력), 출력 512
model.add(Activation('relu'))
model.add(Dropout(0,2))
model.add(Dense(10)) #입력512(이전 레이어 입력), 출력 10
model.add(Activation('softmax'))

#5. 모델 구축하기
model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy'])

#6.데이터 훈련
hist=model.fit(x_train,y_train)

#7.테스트데이터로 평가하기
score=model.evaluate(x_test,y_test,verbose=1)
print('lose=',score[0])
print('accuracy=',score[1])

#BMI
import random
#BMI 를 계산해서 레이블을 리턴하는 함수
def calc_bmi(h,w):
    bmi=w/(h/100)**2
    if bmi < 18.5: return 'thin'
    if bmi < 25: return 'normal'
    return 'fat'
#출력 파일 준비
fp=open('bmi.csv','w',encoding='utf-8')
fp.write('height,weight,label\r\n')

#무작위로 데이터 생성
cnt={'thin':0,'normal':0,'fat':0}
for i in range(20000):
    h=random.randint(120,200)
    w=random.randint(35,80)
    label=calc_bmi(h,w)
    cnt[label]+=1
    fp.write('{0},{1},{2}\r\n'.format(h,w,label))
fp.close()
print('ok',cnt)

#1. data load
import pandas as pd
from sklearn.cross_validation import train_test_split
data=pd.read_csv('bmi.csv')

#2. 데이터 정규화하기
data['weight']/=80
data['height']/=200
x=data[['weight','height']].as_matrix()
bclass={'thin':[1,0,0],'normal':[0,1,0],'fat':[0,0,1]}
y=np.empty((20007,3))
#empty: 초기화 되지 않는 배열

#레이블
#enumerate: 열거
for i,v in enumerate(data['label']):
    print(i,v)

for i,v in enumerate(data['label']):
    y[i]=bclass[v] 
    
#3. split train,test
x_train,x_test=train_test_split(x,test_size=0.2)
y_train,y_test=train_test_split(y,test_size=0.2)

#모델 구조 정의
#sequential model: 레이어들을 선형으로 쌓는 모델
model=Sequential()
model.add(Dense(512,input_shape=(2,))) #입력 2, 출력 512
model.add(Activation('relu'))
model.add(Dropout(0,1))
model.add(Dense(512)) #입력 512(이전 레이어 입력), 출력 512
model.add(Activation('relu'))
model.add(Dropout(0,1))
model.add(Dense(3)) #입력512(이전 레이어 입력), 출력 3
model.add(Activation('softmax'))

#모델 구축하기
model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy'])

#데이터 훈련하기
hist=model.fit(
        x_train,y_train)
hist=model.fit(
        x_train,y_train,
        batch_size=100,
        nb_epoch=20,
        verbose=1)

#테스트 데이터로 평가하기
score=model.evaluate(x_test,y_test)
print('loss=',score[0])
print('accuracy=',score[1])
