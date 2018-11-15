# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 09:37:25 2018
문장 입력 다중클래스분류 모델
참조: https://tykimos.github.io/2017/08/17/Text_Input_Multiclass_Classification_Model_Recipe/
@author: SDEDU
"""

from keras.datasets import reuters
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

#1. 데이터셋 생성
max_features=15000
text_max_words=120

#훈련셋과 시험셋 불러오기
(x_train,y_train),(x_test,y_test)=reuters.load_data(num_words=max_features)

#훈련셋과 검증셋 분리
x_val=x_train[7000:]
y_val=y_train[7000:]
x_train=x_train[:7000]
y_train=y_train[:7000]

#데이터셋 전처리 :문장길이 맞추기
#문장길이를 maxlen인자로 맞춤(=120보다 짧은 문장은 0을 채워서 120단어로 맞춰주고 120보다 긴 문장은 120단어까지 잘라냄)
x_train=sequence.pad_sequences(x_train,maxlen=text_max_words)
x_val=sequence.pad_sequences(x_val,maxlen=text_max_words)
x_test=sequence.pad_sequences(x_test,maxlen=text_max_words)

#one-hot 인코딩 onverts a class vecoter (integers) to binary class matrix
y_train=np_utils.to_categorical(y_train)
y_val=np_utils.to_categorical(y_val)
y_test=np_utils.to_categorical(y_test)


#2. 모델 구성하기
#순환 신경망 모델
model=Sequential()
model.add(Embedding(max_features,128))
model.add(LSTM(128))
model.add(Dense(46,activation='softmax'))

#3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#4. 모델 학습시키기
hist=model.fit(x_train,y_train,epochs=10, batch_size=64,validation_data=(x_val,y_val))

#5. 학습과정 살펴보기
import matplotlib.pyplot as plt
fig,loss_ax=plt.subplots()

acc_ax=loss_ax.twinx()

loss_ax.plot(hist.history['loss'],'y',label='train loss')
loss_ax.plot(hist.history['val_loss'],'r',label='val loss')
loss_ax.set_ylim([0.0,3.0])

acc_ax.plot(hist.history['acc'],'b',label='train acc')
acc_ax.plot(hist.history['val_acc'],'g',label='val acc')
acc_ax.set_ylim([0.0,1.0])

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

#6. 모델평가하기
loss_and_metrics=model.evaluate(x_test,y_test,batch_size=64)
print(loss_and_metrics)


