# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:18:02 2018
Caltech 101 데이터 이용
@author: SDEDU
"""

'''
이미지를 학습할 때마다 원본크기의 이미지를 읽어처리하는 것은 비효율적이다.
Caltech 101 이미지는 크기가 모두 달라서 머신러닝으로 다루기 불편
일정한 크기로 리사이즈, 24비트 RGB 형식으로 변환
NUMPY 배열 형식으로 저장

다운로드: http://www.vision.caltech.edu/Image_Datasets/Caltech101/Caltech101.html#Download
각 카테고리에는 이미지가 약 60장, 전체 337장 사신으로 대상을 분류
'''
from PIL import Image
import os,glob
import numpy as np
from sklearn.model_selection import train_test_split
##서로 다른 크기의 색상 이미지 데이터를 동일한 규격으로 변환하기######
#1. 분류대상 카테고리 선택하기
caltech_dir='C:\\Users\\SDEDU\\Downloads\\101_ObjectCategories'
categories=['chair','camera','butterfly','elephant','flamingo']
nb_classes=len(categories)

#2. 이미지 크기 지정
image_w=64
image_h=64
pixels=image_w * image_h *3

#3. 이미지 데이터 읽어 들이기
x=[]
y=[]
for idx,cat in enumerate(categories):
    label=[0 for i in range(nb_classes)]#레이블 지정
    label[idx]=1
    
    image_dir=caltech_dir + '/' + cat #이미지
    files=glob.glob(image_dir+'/*.jpg')
    for i, f in enumerate(files):
        img=Image.open(f)
        img=img.convert('RGB')
        img=img.resize((image_w,image_h))
        data=np.asarray(img)
        x.append(data)
        y.append(label)
        if i %10 ==0:
            print(i,'\n',data)
x=np.array(x)
y=np.array(y)
#4. 학습 전용 데이터와 테스트 전용 데이터 구분
x_train,x_test,y_train,y_test=train_test_split(x,y)
xy=(x_train,x_test,y_train,y_test)
np.save('5obj.npy',xy)
print('ok',len(y))

##CNN으로 학습하기 ######################
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D
from keras.layers import Activation,Dropout,Flatten,Dense
import h5py #keras로 모델을 저장할 떄는 HDFS 형식의 데이터를 다루는  h5py 모듈사용
from PIL import Image
import numpy as np
import os
#1. 카테고리 저장
categories=['chair','camera','butterfly','elephant','flamingo']
nb_classes=len(categories)
#2. 이미지 크기 저장
image_w=64
image_h=64
#3. 데이터 불러오기
x_train,x_test,y_train,y_test=np.load('5obj.npy')
#4. 데이터 정규화하기
x_train=x_train.astype('float')/255
x_test=x_test.astype('float')/255
print('x_train shape:',x_train.shape)

#6. 모델 구축하기
nfilter=bsize=32
opt=['adam','rmsprop']
model = Sequential()
x_train.shape[1:]
model.add(Convolution2D(32, (3, 3), activation='relu',border_mode='same', input_shape=(64, 64, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(nfilter*2, (3, 3), activation='relu',border_mode='same'))
model.add(Convolution2D(nfilter*2, (3, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))
#7. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer=opt[1], metrics=['accuracy'])

#8. 모델 학습시키기
hist = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_test, y_test))


#9. 모델 훈련하기
hdf5_file='5obj--model.hdf5'
if os.path.exists(hdf5_file):
    model.load_weights(hdf5_file)
else:
    model.fit(x_train,y_train,batch_size=32,nb_epoch=50)
    model.save_weights(hdf5_file)

#10. 모델 평가하기
pre=model.predict(x_test)
for i,v in enumerate(pre):
    pre_ans=v.argmax()
    ans=y_test[i].argmax()
    dat=x_test[i]
    if ans==pre_ans: continue
    print('[NG]', categories[pre_ans], '!=',categories[ans])
    print(v)
    fname='error/'+str(i)+'-'+categories[pre_ans] + '-ne-'+categories[ans] + '.png'
    dat*=256
    img=Image.fromarray(np.uint8(dat))
    img.save(fname)
score=model.evaluate(x_test,y_test)
print('loss=',score[0])
print('acc=',score[1])
# 5. 학습과정 살펴보기

import matplotlib.pyplot as plt
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([0.0, 0.5])

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylim([0.8, 1.0])

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 6. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('## evaluation loss and_metrics ##')
print(loss_and_metrics)

# 7. 모델 사용하기
yhat_test = model.predict(x_test, batch_size=32)


import math
plt_row = 4
plt_col = 4

plt.rcParams["figure.figsize"] = (10,10)

f, axarr = plt.subplots(plt_row, plt_col)

cnt = 0
i = 0

while cnt < (plt_row*plt_col):
    
    if np.argmax(y_test[i]) == np.argmax(yhat_test[i]):
        i += 1
        continue
    
    sub_plt = axarr[math.trunc(cnt/plt_row), cnt%plt_col]
    sub_plt.axis('off')
    sub_plt.imshow(x_test[i])
    sub_plt_title = 'R: ' + str(np.argmax(y_test[i])) + ' P: ' + str(np.argmax(yhat_test[i]))
    sub_plt.set_title(sub_plt_title)

    i += 1    
    cnt += 1

plt.show()


