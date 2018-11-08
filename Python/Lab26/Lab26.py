# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:58:07 2018

@author: SDEDU
"""
'''
머신러닝 딥러닝 차이
머신러닝: 전처리 과정 거치고 돌림(사람이 기계에 학습 시키고 문제를 해결)
딥러닝: 기계가 알아서 다 해라(데이터를 그대로 주고 스스로 기계가 분석 후 답을 내는 방식)

기계학습(머신러닝) : 컴퓨터가 데이터를 분석하고, 문제점을 스스로 고치면서 정보를 추론하는 기술
1)지도학습(분류, 회귀) : 정답을 알려주며 학습시키는 것
분류: 데이터에 대해 두가지중 하나로 분류하는 것
다중분류 : 어떤 데이터에 대해 여러 값 중 하나로 분류하는 것
회귀: 어떤 데이터들의 특징을 토대로 값을 예측하는 것
2)비지도학습 : 정답을 알려주지 않고 비슷한 데이터들을 군집화 하는것
3)강화 학습 : 상과 벌이라는 보상을 주며 상을 최대화하고 벌을 최소화 하도록 강화 학습하는 방식

신경망은 입력층, 은닉층, 출력층 으로 되어있다.
입력층 : 가장 왼쪽 출( 입력이 들어오는 층)
은닉층 : 중간에 모든 층 (사람 눈에 보이지 않는 층이다)
출력층 : 맨 오른쪽 층 (출력되는 층)

b는 편향 : 얼마나 쉽게 활성화되느냐
w는 가중치 :각 신호의 가중치를 나타내는 매개변수 각신호의 영향력을 제어

퍼셉트론 : 두개의 신호를 받아 output을 출력하는 퍼셉트론
output= if w*x+b <= 0   -->  0
        if w*x+b > 0    -->  1
        
단순 퍼셉트론: 단층 네트워크에서 계단 함수를 활성화 함수로 사용한 모델
다중 퍼셉트론: 신경망을 가르킴(= 여러층으로 구성되고 시그모이드 함수 등의 매끈한 활성화 함수를 사용)
XOR 게이트는 다중 퍼셉트론으로 만들 수 있음


시그모이드 함수 h(x) = 1/(1 + exp(-x)) = 1/(1 + e^-x )

순전파: 이미 학습된 매개변수를 사용하여 학습과정을 생략하고,추론과정만을 구현하는 추론과정을 말함

기계학습 문제는 회귀,분류로 나눌 수 있음
출력층의 활성화 함수로 회귀에서는 주로 항등함수,
분류에서는 주로 소프트맥스 함수를 이용
분류에서는 출력층의 뉴런 수로 분류하는 클래스 수와 같게 설정

손실함수: 신경망 학습에서 사용하는 하나의 지표

SIFT : 이미지의 크기와 회전에 불변하는 특징을 추출하는 알고리즘

신경망 학습의 목표: 손실 함수 값이 가장 작아지는 가중치 매개변수 값을 찾아내는 것

참조한 사이트

https://m.blog.naver.com/htk1019/221089905008
http://nbviewer.jupyter.org/github/SDRLurker/deep-learning/blob/master
'''



import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

X=np.array([1.0,0.5]) 
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]]) 
B1=np.array([0.1,0.2,0.3]) 

(W1.shape)
X.shape
B1.shape

A1=np.dot(X,W1)+B1

Z1=sigmoid(A1)

W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2=np.array([0.1,0.2])

A2=np.dot(Z1,W2)+B2
Z2=sigmoid(A2)

#항등 함수
def identity_function(x):
    return x

W3=np.array([[0.1,0.3],[0.2,0.4]])
B3=np.array([0.1,0.2])

A3=np.dot(Z2,W3)+B3
Y=identity_function(A3)

def init_network():
    network={}
    network['w1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1']=np.array([0.1,0.2,0.3])
    network['w2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2']=np.array([0.1,0.2])
    network['w3']=np.array([[0.1,0.3],[0.2,0.4]])
    network['b3']=np.array([0.1,0.2])
    
    return network

def forward(network,x):
    W1,W2,W3=network['w1'],network['w2'],network['w3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    
    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=identity_function(a3)
    
    return y

network=init_network()
x=np.array([1.0,0.5])
y=forward(network,x)
y


'''
소프트맥스 함수의 원래 식대로 할 때 오버플로우가 발생 할 가능성이 있다.
따라서 수정한 식이 쓰인다.
'''
#원래 소프트맥스 함수
a=np.array([0.3,2.9,4.0])
exp_a=np.exp(a) #지수 함수
exp_a

sum_exp_a=np.sum(exp_a) 
sum_exp_a

y=exp_a/sum_exp_a
y

def softmax(a):
    exp_a=np.exp(a)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    
    return y

#수정한 소프트맥스 함수
a=np.array([1010,1000,990])
np.exp(a) #오버플로우

c=np.max(a)
a-c
np.exp(a-c)/np.sum(np.exp(a-c))

def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

a=np.array([0.3,2.9,4.0])
y=softmax(a)
y
np.sum(y) #소프트맥스 함수 출력의 총합은 1 (확률로 해석 가능)


#손글씨
import sys, os
sys.path.append(os.pardir) #부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image

(x_train, t_train),(x_test,t_test)=load_mnist(flatten=True,normalize=False)

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000,)
print(x_test.shape)  # (10000, 784)
print(t_test.shape)  # (10000,)

def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))
    plt.imshow(np.array(pil_img))
    
img=x_train[0]
label=t_train[0]
label #5
img.shape #784,
img=img.reshape(28,28) #원래 이미지의 모양으로 변형
img.shape #28,28

img_show(img)


#신경망의 추론 처리
import pickle
def get_data():
    (x_train,t_train),(x_test,t_test)=load_mnist(normalize=True, flatten=True,one_hot_label=False)
    return x_test,t_test


#pickle 파일인 sample_weight.pkl에 저장된 '학습된 가중치 매개변수'를 읽음
def init_network():
    with open('sample_weight.pkl','rb') as f:
        network=pickle.load(f)        
    return network

def predict(network,x):
    W1, W2, W3 = network['W1'],network['W2'],network['W3']
    b1, b2, b3 = network['b1'],network['b2'],network['b3']
    
    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=softmax(a3)  
    return y

x,t =get_data()
network=init_network()

accuracy_cnt=0
for i in range(len(x)):
    y=predict(network,x[i])
    p=np.argmax(y) #확률이 가장 높은원소의 인덱스
    if p==t[i]:
        accuracy_cnt +=1
        
print('Accuracy:' + str(float(accuracy_cnt)/len(x))) #0.9352


#배치 처리
x,_=get_data()
network=init_network()
W1,W2,W3=network['W1'],network['W2'],network['W3']

x.shape #10000,784
x[0].shape #784,
W1.shape #784,50
W2.shape #50,100
W3.shape #100,10
#최종 결과는 원소가 10개인 1차원 배열 y가 출력됨 

#배치 처리 구현
x,t=get_data()
network=init_network()
batch_size=100 #배치 크기
accuracy_cnt=0
for i in range(0,len(x),batch_size):
    x_batch=x[i:i+batch_size]
    y_batch=predict(network,x_batch)
    p=np.argmax(y_batch,axis=1)
    accuracy_cnt += np.sum(p==t[i:i+batch_size])
print('Accuracy:' + str(float(accuracy_cnt)/len(x)))

#손실 함수 
#1)평균 제곱 오차
def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)
import numpy as np

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] #0.6이 제일 크므로 2번쨰 인덱스만 1
mean_squared_error(np.array(y),np.array(t)) #'2' 일 확률이 가장 높다고 추정함(0.6)

y=[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
mean_squared_error(np.array(y),np.array(t)) #'7' 일 확률이 가장 높다고 추정함(0.6)
# -> 평균 제곱 오차가 첫번째 추정결과가 오차가 더 작으므로 정답에 가까울것이라 판단


import matplotlib.pylab as plt
#자연 로그
x=np.arange(0.001,1.0,0.001)
y=np.log(x)
plt.plot(x,y)
plt.ylim(-5,0)

#2)교차 엔트로피 구현
def cross_entropy_error(y,t):
    delta=1e-7
    return -np.sum(t*np.log(y+delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
cross_entropy_error(np.array(y),np.array(t)) #0.5108..

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
cross_entropy_error(np.array(y),np.array(t)) #2.3025..
# -> 첫번째추정이 정답일 가능성이 높다고 판단

#3)미니배치 학습
train_size=x_train.shape[0]
batch_size=10
batch_mask=np.random.choice(train_size,batch_size)
x_batch=x_train[batch_mask]
t_batch=t_train[batch_mask]


#시그모이드 함수
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_diff(x):
    return sigmoid(x)*(1-sigmoid(x))
def 시그모이드_접선(x): # 접선 ax + b에서 a,b값을리턴 
    return sigmoid_diff(x), sigmoid(x) - sigmoid_diff(x) * x

x=np.arange(-6.0,6.0,0.1)
y1=sigmoid(x)
a2,b2=시그모이드_접선(4)
y2=a2*x + b2
a3,b3 = 시그모이드_접선(-4)
y3=a3*x + b3
plt.plot(x,y1)
plt.plot(x,y2,color='green')
plt.plot(x,y3,color='black')
plt.scatter([4,-4],[a2*4+b2,a3*-4 +b3],color='red')

'''
(base) C:\Users\SDEDU>pip install pystan
(base) C:\Users\SDEDU>pip install pandas_datareader
(base) C:\Users\SDEDU>conda install -c conda-forge fbprophet
Visual C++ Build Tools
'''
import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
from datetime import datetime

path='C:/Windows/Fonts/malgun.ttf'
import platform
from matplotlib import font_manager, rc
if platform.system()=='Darwin':
    rc('font',family='AppleGothic')
elif platform.system()=='Windows':
    font_name=font_manager.FontProperties(fname=path).get_name()
    rc('font',family=font_name)
else:
    print('Unknown system')
    
plt.rcParams['axes.unicode_minus']=False
pinkwink_web=pd.read_csv('08. PinkWink Web Traffic.csv',
                         encoding='utf-8', thousands=',',
                         names=['date','hit'], index_col=0)

pinkwink_web=pinkwink_web[pinkwink_web['hit'].notnull()]
pinkwink_web.head()

pinkwink_web['hit'].plot(figsize=(12,4),grid=True)

