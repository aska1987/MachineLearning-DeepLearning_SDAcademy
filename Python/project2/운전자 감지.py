# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:39:40 2018

@author: SDEDU
"""
'''
코드 5-2
탐색적 데이터 분석을 위해 필요한 라이브러리와 함수 정의하기
'''

import os 
from glob import glob
#이미지 데이터를 다루는 open CV 라이브러리
import cv2
#시각화 관련 라이브러리
import matplotlib.pyplot as plt

def read_image(path):
    #OpenCV는 이미지 데이터를 B(blue), G(Green), R(red) 순서로 읽어오기 때문에,
    #cv2.cvtColor() 함수를 통해 R,G,B 순서로 변경한다.
    image=cv2.imread(path,cv2.IMREAD_COLOR)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image

'''
코드 5-3
클래스 c0의 예시 이미지(img_100025.jpg)를 시각화하는 코드
'''
#이미지 파일 경로를 지정
data_dir='C://Users/SDEDU/.kaggle/competitions/state-farm-distracted-driver-detection/'
train_path=data_dir + 'imgs/train/c0/'
filename='img_100026.jpg'

#이미지 데이터 읽어오기
image=read_image(train_path + filename)

#이미지 시각화
plt.imshow(image)

'''
코드 5-4
클래스별로 9개의 임의의 이미지를 시각화하기
'''
#훈련 데이터 클래스별 예시를 시각화
labels=['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
col_to_kor={
        'c0': '안전 운전',
        'c1': '오른손으로 문자',
        'c2': '오른손으로 전화',
        'c3': '왼손으로 문자',
        'c4': '왼손으로 전화',
        'c5': '라디오 조작',
        'c6': '음료수 섭취',
        'c7': '뒷자석에 손 뻗기',
        'c8': '얼굴, 머리 만지기',
        'c9': '조수석과 대화'
            }

for label in labels:
    f,ax = plt.subplots(figsize=(12,10))
    files=glob('{}/imgs/train/{}/*.jpg'.format(data_dir,label))
    
    #총 9개의 이미지를 시각화 한다
    for x in range(9):
        plt.subplot(3,3,x+1)
        image=read_image(files[x])
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    print('\t\t\t\t# {} : {}'.format(label,col_to_kor[label]))
          
'''
코드 5-5
테스트 데이터에 포함된 임의의 이미지 18개를 시각화하기
'''
#테스트 데이터 예시를 시각화 한다
f,ax=plt.subplots(figsize=(24,10))
files=glob('{}/imgs/test/*.jpg'.format(data_dir))

#총 18개의 이미지를 시각화한다
for x in range(18):
    plt.subplot(3,6,x+1)
    image=read_image(files[x])
    plt.imshow(image)
    plt.axis('off')
    
'''
코드 5-6
운전자 정보가 담겨진 파일(driver_imgs_list.csv) 읽기
'''
import pandas as pd
#파일 읽기
driver_list=pd.read_csv('C:\\Users\SDEDU\.kaggle\competitions\state-farm-distracted-driver-detection\driver_imgs_list.csv\driver_imgs_list.csv')
#파일의 첫 5줄을 출력
driver_list.head()

'''
코드 5-7
운전자 ID 고유값의 개수를 구하기
'''
import numpy as np
len(np.unique(driver_list['subject']).tolist())

'''
코드 5-8
운전자별로 훈련 데이터를 시각화하기
'''
#운전자별 이미지 데이터를 저장하는 dict를 생성
driver_to_img={}
for i,row in driver_list.iterrows():
    driver=row['subject']
    label=row['classname']
    image_path=row['img']
    if not driver_to_img.get(driver,False):
        driver_to_img[driver]=[image_path]
    else:
        driver_to_img.get(driver).append(image_path)
#운전자별 훈련 데이터 예시를 시각화한다
for driver in np.unique(driver_list['subject']).tolist():
    for label in labels:
        f,ax=plt.subplots(figsize=(12,10))
        files=glob('{}/imgs//train/{}/*.jpg'.format(data_dir,label))
        print_files=[]
        for fl in files:
            if(driver_list[driver_list['img']==os.path.basename(fl)]
            ['subject']==driver).values[0]:
                print_files.append(fl)
                
        #총 9개의 이미지를 시각화한다
        for x in range(9):
            plt.subplot(3,3,x+1)
            image=read_image(print_files[x])
            plt.imshow(image)
            plt.axis('off')
        plt.show()
        
        #운전자 ID 와 클래스를 출력
        print('\t\t\t\t#운전자 : {} | 클래스 : "{} : {}"'.format(driver,
              label,col_to_kor[label]))
              
'''
코드 5-9
클래스 c0에  속하는 특이한 훈련 데이터(outliers)를 시각화하기
c0 : 안전운전 으로 분류된 outlier 이미지
'''
#Label : c0 안전운전
label='c0'
imgs=[21155,31121]
print('# "c0:안전운전"outliers')
f,ax=plt.subplots(figsize=(12,10))
for x in range(len(imgs)):
    plt.subplot(1,2,x+1)
    image=read_image('{}imgs/train/{}/img_{}.jpg'.format(data_dir,label,imgs[x]))
    plt.imshow(image)
    plt.axis('off')
plt.show()

'''
코드 5-10
클래스 c3에 속하는 특이한 훈련 데이터(Outliers)를 시각화하기
'왼손으로 문자' 로 분류된 outlier 이미지
'''
#Label:c3 왼손으로 문자
label='c3'
imgs=[38563,45874,49269,62784]
print('# "c3: 왼손으로 문자" outliers')
      
f,ax=plt.subplots(figsize=(12,10))
for x in range(len(imgs)):
    plt.subplot(2,2,x+1)
    image=read_image('{}imgs/train/{}/img_{}.jpg'.format(data_dir,label,imgs[x]))
    plt.imshow(image)
    plt.axis('off')
plt.show()

'''
코드 5-11
클래스 c4에 속하는 특이한 훈련 데이터(Outliers)를 시각화하기
왼손으로 전화로 분류된 아웃라이어 이미지
'''
#Label:c4 왼손으로 전화
label='c4'
imgs=[92769,38427,41743,69998,77347,16077]
print('# "c4: 왼손으로 전화" outliers')
f,ax=plt.subplots(figsize=(18,10))
for x in range(len(imgs)):
    plt.subplot(2,3,x+1)
    image=read_image('{}imgs/train/{}/img_{}.jpg'.format(data_dir,label,imgs[x]))
    plt.imshow(image)
    plt.axis('off')
plt.show()

'''
코드 5-12
클래스 c9에 속하는 특이한 훈련 데이터(outliers)를 시각화하기
조수석과 대화로 분류된 아잇라이어 이미지
'''
#label : c9 조수석과 대화
label='c9'
imgs=[28068,37708,73663]
print('# "c9:조수석과 대화" outliers')
f,ax=plt.subplots(figsize=(18,10))
for x in range(len(imgs)):
    plt.subplot(1,3,x+1)
    image=read_image('{}imgs/train/{}/img_{}.jpg'.format(data_dir,label,imgs[x]))
    plt.imshow(image)
    plt.axis('off')
    
'''
코드 5-13
클래스 c0에 속하는 잘못 분류된 데이터를 시각화하기
,다른 클래스로 분류된 c0 이미지
'''
imgs=[('c5',30288),('c7',46617),('c8',3835)]
f,ax=plt.subplots(figsize=(18,10))
print('Examples of c0 : 안전운전 classified in wrong labels')
for x in range(len(imgs)):
    plt.subplot(1,3,x+1)
    image=read_image('{}imgs/train/{}/img_{}.jpg'.format(data_dir,imgs[x][0],
                     imgs[x][1]))
    plt.imshow(image)
    plt.axis('off')
plt.show()

'''
코드 5-14
클래스 c1에 속하는 오분류 데이터를 시각화하기
다른 클래스로 분류된 c1 이미지
'''
#Real Label : c1
imgs=[('c0',29923),('c0',79819),('c2',32934)]
f,ax=plt.subplots(figsize=(18,10))
print('Examples of c1: 오른손으로 문자 classified in wrong labels')
for x in range(len(imgs)):
    plt.subplot(1,3,x+1)
    image=read_image('{}/imgs/train/{}/img_{}.jpg'.format(data_dir,imgs[x][0],
                     imgs[x][1]))
    plt.imshow(image)
    plt.axis('off')
plt.show()

'''
코드 5-15
클래스 C8에 속하는 잘못 분류된 데이터를 시각화하기
다른 클래스로 분류된 c8 이미지
'''
#Real Label : c8
imgs=[('c0',34380),('c3',423),('c5',78504)]
f,ax=plt.subplots(figsize=(18,10))

print(' Examples of c8 : 얼굴,머리만지기 classified in wrong labels')
for x in range(len(imgs)):
    plt.subplot(1,3,x+1)
    image=read_image('{}imgs/train/{}/img_{}.jpg'.format(data_dir,
                     imgs[x][0],imgs[x][1]))
    plt.imshow(image)
    plt.axis('off')
plt.show()

'''
데이터 어그멘테이션: 이미지 데이터에 랜덤한 노이즈를 추가하여 
모델이 학습에 사용하는 데이터를 인위적으로 부풀리는 방법
딥러닝모델입장에서는 훈련데이터의 양이 대폭 증가한 것과 동일한 효과를 얻을 수 있음
'''
'''
코드 5-16
데이터 어그멘테이션을 수행할 예시 이미지 3개의 원본을 시각화하기
'''
#이미지 파일 경로를 지정
img_path=[('c0',55301),('c5',92551),('c8',71055)]

#이미지를 그대로 읽어온다
imgs=[]
for x in range(len(img_path)):
    imgs.append(read_image('{}imgs/train/{}/img_{}.jpg'.format(data_dir,
                           img_path[x][0],img_path[x][1]))/255.)
#이미지 시각화
f,ax=plt.subplots(figsize=(18,10))
for i, img in enumerate(imgs):
    plt.subplot(1,3,i+1)
    plt.imshow(img)
    plt.axis('off')
plt.show()

'''
코드 5-17
임의의 회전 각도로 회전한 이미지를 시각화하기
'''
from scipy.ndimage import rotate
#임의의 회전 각도를 구한 후, 이미지를 회전한다.
rotate_angle=np.random.randint(40)-20
print('# 이미지 회전: {}도'.format(rotate_angle))
for i, img in enumerate(imgs):
    imgs[i]=rotate(img,rotate_angle)
    
#이미지를 시각화
f,ax=plt.subplots(figsize=(18,10))
for x,img in enumerate(imgs):
    plt.subplot(1,3,x+1)
    plt.imshow(img)
    plt.axis('off')
plt.show()

'''
코드 5-18
임의의 확대 비율로 확대한 이미지를 시각화하기
'''
def crop_center(img,cropx,cropy):
    #이미지 중간을 crop 하는 함수를 정의한다
    y,x=img.shape
    startx=x//2-(cropx//2)
    starty=y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

#x, y 축의 이미지 확대 비율을 랜덤으로 정의한다
width_zoom=int(image.shape[0]*(0.8 + 0.2 * (1- np.random.random())))
height_zoom=int(image.shape[1]* (0.8 + 0.2 *(1-np.random.random())))

#이미지를 확대한다
print('# 이미지 확대: (x:{},y:{})'.format(round(1. *width_zoom/image.shape[0],2),
      round(1. * height_zoom/image.shape[1],2)))
for i, img in enumerate(imgs):
    final_image=np.zeros((width_zoom, height_zoom,3))
    final_image[:,:,0]=crop_center(img[:,:,0], height_zoom,width_zoom)
    final_image[:,:,1]=crop_center(img[:,:,1], height_zoom,width_zoom)
    final_image[:,:,2]=crop_center(img[:,:,2], height_zoom,width_zoom)
    imgs[i]=final_image
#이미지 시각화
f,ax=plt.subplots(figsize=(18,10))
for x, img in enumerate(imgs):
    plt.subplot(1,3,x+1)
    plt.imshow(img)
    plt.axis('off')
plt.show()

'''
코드 5-19
임의의 커널 크기로 흐린 이미지를 시각화하기
'''
# 10x10 크기의 커널로 이미지를 흐린다
blur_degree=10
print('{}x{} 커널 크기로 이미지 흐리기'.format(blur_degree,blur_degree))
for i, img in enumerate(imgs):
    imgs[i]=cv2.blur(img,(blur_degree,blur_degree))
#이미지 시각화
f,ax=plt.subplots(figsize=(18,10))
for x, img in enumerate(imgs):
    plt.subplot(1,3,x+1)
    plt.imshow(img)
    plt.axis('off')
plt.show()

'''
1. CNN 모델정의
VGG16: 이미지 분류 모델의 대표격
코드 5-20
VGG16 모델과 Optimizer를 정의하고 학습 가능한 모델로 컴파일하는 코드
'''
import keras
from keras.models import Model
from keras.layers.core import Dense,Dropout,Flatten
from keras.optimizers import SGD
def get_model():
    #최상위 전결층을 제외한 vgg16 모델을 불러온다
    base_model=keras.applications.vgg16.VGG16(include_top=False,
                                              weights=None, input_shape=(224,224,3))
    #최상위 전결층을 정의한다
    out=Flatten()(base_model.output)
    out=Dense(2048,activation='relu')(out)
    out=Dropout(0.5)(out)
    out=Dense(2048,activation='relu')(out)
    out=Dropout(0.5)(out)
    output=Dense(10,activation='softmax')(out)
    model=Model(inputs=base_model.input, outputs=output)
    
    #SGD Optimizer를 사용하여 모델을 compile 한다
    sgd=SGD(lr=le-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.complie(optimizer=sgd,loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
'''
2. 데이터 전처리
코드 5-21
데이터를 실시간으로 읽어오는 ImageDataGenerator() 정의
'''
from keras.preprocessing.image import ImageDataGenerator
#이미지 데이터 전처리를 수행하는 함수를 정의한다
datagen=ImageDataGenerator()
#flow_from_directory() 함수를 통해 특정 폴더에 위치해있는 훈련/검증 데이터를 실시간으로 일겅온다
train_generator=datagen.flow_from_directory(
        directory='../input/train',
        target_size=(224,224),
        batch_size=8,
        class_mode='categorical',
        seed=2018)
valid_generator=datagen.flow_from_directory(
        directtory='../input/valid',
        target_size=(224,224),
        batch_size=8,
        class_mode='categorical',
        seed=2018)
#테스트 데이터 예측용 데이터 생성기를 정의한다
test_generator=datagen.flow_from_directory(
        directory='../input/test',
        target_size=(224,224),
        batch_size=1,
        class_mode=None,
        shuffle=False)
