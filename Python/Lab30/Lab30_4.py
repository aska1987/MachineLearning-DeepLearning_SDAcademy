# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:51:03 2018
시계열 데이터를 예측하는 LSTM 구현
세계 항공 여행 승객 수 예측
@author: SDEDU
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def create_dataset(dataset,look_back=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-look_back):
        a=dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i+look_back,0])
    return np.array(dataX),np.array(dataY)

np.random.seed(7)
data=pd.read_csv('international-airline-passengers.csv',usecols=[1])
data=data[:144]
dataset=data.values
dataset=dataset.astype('float32')

#normalize
scaler=MinMaxScaler(feature_range=(0,1))
dataset=scaler.fit_transform(dataset)

train_size=int(len(dataset)*0.67)
test_size=len(dataset)-train_size
train,test=dataset[0:train_size,:],dataset[train_size:len(dataset),:]

#reshape into x=t, y=t+1
look_back=1
trainX,trainY=create_dataset(train,look_back)
testX,testY=create_dataset(test,look_back)

#reshape input to be [samples,time steps, features]
trainX=np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))
testX=np.reshape(testX,(testX.shape[0],1,testX.shape[1]))

#create and fit the LSTM network
model=Sequential()
model.add(LSTM(4,input_shape=(1,look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(trainX,trainY,epochs=100,batch_size=1,verbose=2)

trainPredict=model.predict(trainX)
testPredict=model.predict(testX)

#invert 
trainPredict=scaler.inverse_transform(trainPredict)
trainY=scaler.inverse_transform([trainY])
testPredict=scaler.inverse_transform(testPredict)
testY=scaler.inverse_transform([testY])

#calculate root mean squared error
trainScore=math.sqrt(mean_squared_error(trainY[0],trainPredict[:,0]))
print('Train Scroe:',trainScore)
testScore=math.sqrt(mean_squared_error(testY[0],testPredict[:,0]))
print('Test Score:',testScore)

#shift train predictions for plotting
trainPredictPlot=np.empty_like(dataset)
trainPredictPlot[:,:]=np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back,:]=trainPredict
#shift test predictions for plotting
testPredictPlot=np.empty_like(dataset)
testPredictPlot[:,:]=np.nan
testPredictPlot[len(trainPredict)+(look_back*2):len(dataset),:]=testPredict

#plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()