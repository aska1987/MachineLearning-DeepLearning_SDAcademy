# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:33:54 2018

tensorflow 
MNIST

@author: SDEDU
"""
## 입력 인자가 kor, math 2개 인 경우 예측 ##############################
'''
이름 국어공부시간 수학공부시간 점수합계
철수  5시간       5시간       101점
영희  7시간       7시간       141점
민수  8시간       8시간       예측
'''
#가설
#hypothesis = x1*w1 +x2*w2 +b

import tensorflow as tf
#입력, 형태만 알려주는 placeholder 로 정의
x1=tf.placeholder(tf.float32,shape=[None])
x2=tf.placeholder(tf.float32,shape=[None])

#출력, 형태만 알려주는 placeholder 로 정의
y=tf.placeholder(tf.float32,shape=[None])

#변수 Weight, Bias 훈련중에 Tensorflow 내부에 학습 (갱신)
W1 = tf.Variable(tf.random_normal([1]),name='weight1')
W2 = tf.Variable(tf.random_normal([1]),name='weight2')
b = tf.Variable(tf.random_normal([1]),name='bias')

#가설식 정의
hypothesis = x1 * W1 + x2 * W2 +b

#cost 함수정의 ( 오차 함수 : 정답과 얼마나 차이가 있느냐)
cost=tf.reduce_mean(tf.square(hypothesis - y))

#최적화 함수 정의
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)

#그래프 실행준비
sess=tf.Session()
sess.run(tf.global_variables_initializer())

#그래프 실행
for step in range(5001):
    cost_val, W_val1, W_val2, b_val, _ = sess.run([cost,W1,W2,b,train],
                                                  feed_dict={x1:[5,7],x2:[5,7],y:[101,141]})
    if step%500 == 0:
        print(step,cost_val, W_val1, W_val2, b_val)
        
#입력 x를 주고 예측 y를 받아 화면 출력
print('예측 Y: ',
      sess.run(hypothesis,
               feed_dict={x1:[5,7,8],x2:[5,7,8]}))


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
## 가공 데이터 살펴보기 ###
data_file=pd.read_csv('MNIST/mnist_train_100.csv',header=None)
data_file.shape

#이미지 규격이 28x28임 , 첫번째 이미지 확인해보기
image_array=np.asfarray(data_file.loc[0,1:]).reshape((28,28))
plt.imshow(image_array,cmap='Greys',interpolation='None')


#MNIST 데이터 다운
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST/',one_hot=True)

#placeholder, variables 을 설정
#None 자리에 나중에 몇개 할 지 임의로 정함
# 784 는 동시에 input되는 값이 784개 (28 x 28)
x=tf.placeholder(tf.float32,[None,784])
#출력값 = 0 ~ 9 ---> 총 10개
y=tf.placeholder(tf.float32,[None,10])
#변수 (행렬 곱을 위해서 784,10) ex) 100 x 784 행렬 곱 784 x 10  = 100 x 10 
W=tf.Variable(tf.random_normal([784,10])) # => random_normal([x의 2번째, y의 2번째])
b=tf.Variable(tf.random_normal([10]))
logit_y=tf.matmul(x,W)+b

#softmax함수 : 분류할 때 사용하는 함수 (전체가 1이됨) y_k=exp(a_k)/∑(exp(a_i)) = 1개 / 전체
softmax_y=tf.nn.softmax(logit_y)


#cross-Entropy 함수 : Cost 오차를 측정하는 함수 ( 정답 일때 0에 가깝, 오답 일때 무한대에 가깝)
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y*tf.log(softmax_y),
                                             reduction_indices=[1]))

#최적화 함수
optimizer=tf.train.GradientDescentOptimizer(0.1)
train=optimizer.minimize(cross_entropy)

#학습
sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(8000):
    batch_xs,batch_ys=mnist.train.next_batch(1000)
    sess.run(train, feed_dict={x:batch_xs,y:batch_ys})
    if i%1000==0:
        print(i)

#학습된 모델이 정확한지를 계산하고 출력
correct_prediction= tf.equal(tf.arg_max(softmax_y,1),
                             tf.arg_max(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,
                                tf.float32))

#정확도
print(sess.run(accuracy,
               feed_dict={x:mnist.test.images,
                          y:mnist.test.labels}))

