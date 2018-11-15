# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 11:45:05 2018
시계열수치 입력 수치예측 모델 
참조: https://tykimos.github.io/2017/09/09/Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe/
@author: SDEDU
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#convert an array of values into a dataset matrix
def create_dataset(signal_data,look_back=1):
    dataX,dataY=[],[]
    for i in range(len(signal_data)-look_back):
        dataX.append(signal_data[i:(i+look_back),0])
        dataY.append(signal_data[i+look_back,0])
    return np.array(dataX),np.array(dataY)
###다중 퍼셉트론 모델 ##########################
#look_back 인자에 따라 모델의 성능이 달라짐
look_back=40

#1. 데이터 생성
signal_data=np.cos(np.arange(1600)*(20*np.pi/1000))[:,None]
# 데이터 전처리
scaler=MinMaxScaler(feature_range=(0,1))
signal_data=scaler.fit_transform(signal_data)

# 데이터 분리
train=signal_data[0:800]
val=signal_data[800:1200]
test=signal_data[1200:]

# 데이터셋 생성
x_train,y_train=create_dataset(train,look_back)
x_val,y_val=create_dataset(val,look_back)
x_test,y_test=create_dataset(test,look_back)

# 데이터셋 전처리
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_val=np.reshape(x_val,(x_val.shape[0],x_val.shape[1],1))
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

x_train=np.squeeze(x_train)
x_val=np.squeeze(x_val)
x_test=np.squeeze(x_test)

#2. 모델 구성하기
model=Sequential()
model.add(Dense(32,input_dim=40,activation='relu'))
model.add(Dropout(0.3))
for i in range(2):
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.3))
model.add(Dense(1))

#3. 모델 학습과정 설정
model.compile(loss='mean_squared_error',optimizer='adagrad')

#4. 모델 학습
hist=model.fit(x_train,y_train,epochs=200,batch_size=32,validation_data=(x_val,y_val))

#5. 학습과정 살펴보기
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylim(0.0,0.15)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'],loc='upper left')
plt.show()

#6. 모델 평가하기
trainScore=model.evaluate(x_train,y_train,verbose=0)
print('Train Score:',trainScore)
valScore=model.evaluate(x_val,y_val,verbose=0)
print('Validation Score:',valScore)
testScore=model.evaluate(x_test,y_test,verbose=0)
print('Test Score:',testScore)

#7. 모델 사용하기
look_ahead=250
xhat=x_test[0,None]
predictions=np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction=model.predict(xhat,batch_size=32)
    predictions[i]=prediction
    xhat=np.hstack([xhat[:,1:],prediction])
    
plt.figure(figsize=(12,5))
plt.plot(np.arange(look_ahead),predictions,'r',label='prediction')
plt.plot(np.arange(look_ahead),y_test[:look_ahead],label='test function')
plt.legend()
plt.show()

##순환신경망 모델 #############################
# 1. 데이터셋 생성하기
signal_data = np.cos(np.arange(1600)*(20*np.pi/1000))[:,None]

# 데이터 전처리
scaler = MinMaxScaler(feature_range=(0, 1))
signal_data = scaler.fit_transform(signal_data)

# 데이터 분리
train = signal_data[0:800]
val = signal_data[800:1200]
test = signal_data[1200:]

# 데이터셋 생성
x_train, y_train = create_dataset(train, look_back)
x_val, y_val = create_dataset(val, look_back)
x_test, y_test = create_dataset(test, look_back)

# 데이터셋 전처리
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 2. 모델 구성하기
model = Sequential()
model.add(LSTM(32, input_shape=(None, 1)))
model.add(Dropout(0.3))
model.add(Dense(1))

# 3. 모델 학습과정 설정하기
model.compile(loss='mean_squared_error', optimizer='adam')

# 4. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_val, y_val))

# 5. 학습과정 살펴보기
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylim(0.0, 0.15)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# 6. 모델 평가하기
trainScore = model.evaluate(x_train, y_train, verbose=0)
model.reset_states()
print('Train Score: ', trainScore)
valScore = model.evaluate(x_val, y_val, verbose=0)
model.reset_states()
print('Validataion Score: ', valScore)
testScore = model.evaluate(x_test, y_test, verbose=0)
model.reset_states()
print('Test Score: ', testScore)

# 7. 모델 사용하기
look_ahead = 250
xhat = x_test[0]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = model.predict(np.array([xhat]), batch_size=1)
    predictions[i] = prediction
    xhat = np.vstack([xhat[1:],prediction])
    
plt.figure(figsize=(12,5))
plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
plt.plot(np.arange(look_ahead),y_test[:look_ahead],label="test function")
plt.legend()
plt.show()


##상태유지 순환 신경망 모델 #################################
import keras
class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []
        
    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

look_back = 40

# 1. 데이터셋 생성하기
signal_data = np.cos(np.arange(1600)*(20*np.pi/1000))[:,None]

# 데이터 전처리
scaler = MinMaxScaler(feature_range=(0, 1))
signal_data = scaler.fit_transform(signal_data)

# 데이터 분리
train = signal_data[0:800]
val = signal_data[800:1200]
test = signal_data[1200:]

# 데이터셋 생성
x_train, y_train = create_dataset(train, look_back)
x_val, y_val = create_dataset(val, look_back)
x_test, y_test = create_dataset(test, look_back)

# 데이터셋 전처리
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 2. 모델 구성하기
model = Sequential()
model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True))
model.add(Dropout(0.3))
model.add(Dense(1))

# 3. 모델 학습과정 설정하기
model.compile(loss='mean_squared_error', optimizer='adam')

# 4. 모델 학습시키기
custom_hist = CustomHistory()
custom_hist.init()

for i in range(200):
    model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False, callbacks=[custom_hist], validation_data=(x_val, y_val))
    model.reset_states()

# 5. 학습과정 살펴보기
plt.plot(custom_hist.train_loss)
plt.plot(custom_hist.val_loss)
plt.ylim(0.0, 0.15)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# 6. 모델 평가하기
trainScore = model.evaluate(x_train, y_train, batch_size=1, verbose=0)
model.reset_states()
print('Train Score: ', trainScore)
valScore = model.evaluate(x_val, y_val, batch_size=1, verbose=0)
model.reset_states()
print('Validataion Score: ', valScore)
testScore = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
model.reset_states()
print('Test Score: ', testScore)

# 7. 모델 사용하기
look_ahead = 250
xhat = x_test[0]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = model.predict(np.array([xhat]), batch_size=1)
    predictions[i] = prediction
    xhat = np.vstack([xhat[1:],prediction])
    
plt.figure(figsize=(12,5))
plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
plt.plot(np.arange(look_ahead),y_test[:look_ahead],label="test function")
plt.legend()
plt.show()