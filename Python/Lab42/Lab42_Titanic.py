# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 17:45:35 2018

@author: SDEDU
"""

import numpy as np
import pandas as pd
import tensorflow as tf
train_data=pd.read_csv('Titanic/train.csv')
test_data=pd.read_csv('Titanic/test.csv')
test_sub=pd.read_csv('Titanic/gender_submission.csv')

train_data=train_data.as_matrix()
test_data=test_data.as_matrix()
test_sub=test_sub.as_matrix()

#데이터 전처리
#male -> 1
#female -> 0
for i in range(len(train_data)):
    if train_data[i,4]=='male':
        train_data[i,4] =1
    else:
        train_data[i,4]=0
        
train_data[:,4]

for i in range(len(test_data)):
    if test_data[i,3] =='male':
        test_data[i,3]=1
    else:
        test_data[i,3]=0

#승선항 전처리 S->1, C->2, Q->3
for i in range(len(train_data)):
    if train_data[i,11] =='S':
        train_data[i,11]=1
    elif train_data[i,11] =='C':
        train_data[i,11]=2
    elif train_data[i,11] =='Q':
        train_data[i,11]=3
    if np.isnan(train_data[i,11]):
        train_data[i,11]=0
        
        
for i in range(len(test_data)):
    if test_data[i,10] =='S':
        test_data[i,10]=1
    elif test_data[i,10] =='C':
        test_data[i,10]=2
    elif test_data[i,10] =='Q':
        test_data[i,10]=3
    if np.isnan(test_data[i,10]):
        test_data[i,10]=0
        
        
x_passengerData=train_data[:,[2, #Pclass
                              4, #Sex
                              6, #SibSp
                              7, #Parch
                              11 #Embarked
                              ]]
y_survived=train_data[:,1:2] #Survived

        
Test_x_passengerData=test_data[:,[1, #Pclass
                              3, #Sex
                              5, #SibSp
                              6, #Parch
                              10 #Embarked
                              ]]
Test_y_survived=test_sub[:,1:2] #Survived

#placeholder
x=tf.placeholder(tf.float32,shape=[None,5])
y=tf.placeholder(tf.float32,shape=[None,1])

#Variable
W=tf.Variable(tf.random_normal([5,1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

#hypothesis sigmoid를 이용해 0-1 값으로 변경
hypothesis=tf.sigmoid(tf.matmul(x,W)+b)       

#cost, loss 결과값과 예측값의 차이: 작게 만드는 것이 학습
#제곱오차, Cross_Entropy
# Y=1, hypothesis=1  정답 -> tf.log(1) = 0에 가까워짐
# Y=1, hypothesis=0  오답 -> tf.log(0) = 무한대에 가까워짐
# Y=0, hypothesis=1  오답 -> tf.log(0) = 무한대에 가까워짐
# Y=0, hypothesis=0  정답 -> tf.log(1) = 0에 가까워짐
cost=-tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

#optimizer
train=tf.train.GradientDescentOptimizer(0.1).minimize(cost)


predicted=tf.cast(hypothesis>0.5 , dtype=tf.float32)
accuracy=tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32))

#학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10000):
        cost_val,_=sess.run([cost,train],
                            feed_dict={x:x_passengerData,
                                       y:y_survived})
        if step%500==0:
            print('step=',step,'cost=',cost_val)
#훈련데이터로 확인
    h,c,a=sess.run([hypothesis,predicted,accuracy],
                   feed_dict={x:x_passengerData,
                              y:y_survived})
    print('Accuracy: ',a)

#테스트데이터로 확인
    print('Test CSV runningResult')
    h2,c2,a2=sess.run([hypothesis,predicted,accuracy],
                      feed_dict={x:Test_x_passengerData,
                                 y:Test_y_survived})
    print('Accuracy: ',a2)

