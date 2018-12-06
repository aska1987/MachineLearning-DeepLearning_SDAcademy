# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 16:30:29 2018

@author: SDEDU
"""
    
'''
tensorflow 의 자료형
1) Constant : 바로 값이 대입, 정해진값
2) Variable : 내부에서 계속 변경하는 값 (Weight,bias) , global_variable_initializer() 시작할 때 초기화
3) Placeholder : 설계를 한다. 그런데 데이터가 여러 종류 1번데이터를 만개, 다음번에는 2번 데이터를 5만개
training_data, test_data, vaildation_data
'''

#1) 선 디자인 후 실행
import tensorflow as tf
#Build a graph
a=tf.constant(5.0)
b=tf.constant(6.0)
c=a*b
print(c)

#Launch the graph in a session
sess=tf.Session()
print('sess.run(c)=',sess.run(c))

x=tf.constant([[1.,1.],[2.,2.]]) 

sess=tf.Session()
print('Rnak=',sess.run(tf.rank(x)))
print(sess.run(tf.reduce_mean(x)))
print(sess.run(tf.reduce_mean(x,axis=0)))
print(sess.run(tf.reduce_mean(x,axis=1)))

#y=1.4x + 3.5
#학습데이터(x,y)
x_train=[1,2,3,4]
y_train=[6,5,7,10]

#변수 선언
#텐서플로 내부에서 계속 조정하는 값은 Variable로 선언
#가중치(Weight), 편향(Bias)
W=tf.Variable(tf.random_normal([1]),name='weight2')
b=tf.Variable(tf.random_normal([1]),name='bias')

#가설식
hypothesis=x_train * W +b

#cost/loss function
#제곱오차
cost=tf.reduce_mean(tf.square(hypothesis - y_train))

#최적화 함수
#경사하강법
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)
print(train)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

#최적값 찾기 Training
for step in range(2000):
    sess.run(train)
    print(step,sess.run(cost),sess.run(W),sess.run(b))
    

#Build a graph
a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
c=tf.multiply(a,b)
print(c)

sess=tf.Session()
print(sess.run(c,feed_dict={a:3,b:4}))

## 국어 성적 예측하기 ###############################################
# 이름 공부시간 점수
# 철수 5시간    52점
# 영희 7시간    72점
# 민수 8시간    (예측하기)

#입력 , 형태만 알려주는 placeholder로 정의
x=tf.placeholder(tf.float32,shape=[None])
#출력 , 형태만 알려주는 placeholder로 정의
y=tf.placeholder(tf.float32,shape=[None])

#변수 Weight,Bias 정의 초기값은 랜덤값
W=tf.Variable(tf.random_normal([1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

#가설식 정의
hypothesis=x*W +b
#cost 함수 정의
cost=tf.reduce_mean(tf.square(hypothesis-y))
#최적화 함수 정의
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)

#그래프 실행 준비
sess=tf.Session()
sess.run(tf.global_variables_initializer())

#그래프 실행, 500번 마다 화면출력
for step in range(5001):
    cost_val,W_val,b_val,_= sess.run([cost,W,b,train],
                                     feed_dict={x:[5,7],y:[52,72]})
    if step%500 ==0:
        print(step,cost_val,W_val,b_val)
        
#입력X를 주소 예측 y를 받아 화면 출력
print('예측 y:',sess.run(hypothesis,feed_dict={x:[8]}))

## 입력 인자가 kor, math 2개 인 경우 예측 ##############################
'''
이름 국어공부시간 수학공부시간 점수합계
철수  5시간       5시간       101점
영희  7시간       7시간       141점
민수  8시간       8시간       예측
'''
#가설
#hypothesis = x1*w1 +x2*w2 +b
