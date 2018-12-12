# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 12:08:04 2018

@author: SDEDU
"""

'''
MNIST 데이터를 input
중간 게층을 줄이고 
MNIST 데이터를 output으로 복원
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#MNIST 다운
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('tmp/data', one_hot=True)

#학습에 필요한 설정값을 정의
learning_rate=0.02 # 학습률
training_epoch=50  # 반복횟수
batch_size=256     # 배치 개수
display_step=1     # step 정보 출력
examples_to_show=10# MNIST 10개 시각화
input_size=784     # 28 * 28
hidden1_size=256   # 
hidden2_size=128   # 

#신경망 설계
x=tf.placeholder(tf.float32,shape=[None,input_size])

#AutoEncoder 구조 정의
def build_autoencoder(x): # 784 -> 256 -> 128 -> 256 -> 784
    #encoding 784 -> 256 -> 128
    W1=tf.Variable(tf.random_normal(shape=[input_size,hidden1_size]))
    b1=tf.Variable(tf.random_normal(shape=[hidden1_size]))
    H1_output=tf.nn.sigmoid(tf.matmul(x,W1)+b1)
    
    W2=tf.Variable(tf.random_normal(shape=[hidden1_size,hidden2_size]))
    b2=tf.Variable(tf.random_normal(shape=[hidden2_size]))
    H2_output=tf.nn.sigmoid(tf.matmul(H1_output,W2)+b2)
    
    #decoding 128 -> 256 -> 784
    W3=tf.Variable(tf.random_normal(shape=[hidden2_size,hidden1_size]))
    b3=tf.Variable(tf.random_normal(shape=[hidden1_size]))
    H3_output=tf.nn.sigmoid(tf.matmul(H2_output,W3)+b3)
    
    W4=tf.Variable(tf.random_normal(shape=[hidden1_size,input_size]))
    b4=tf.Variable(tf.random_normal(shape=[input_size]))
    reconstructed_x=tf.nn.sigmoid(tf.matmul(H3_output,W4)+b4)
    return reconstructed_x
    
def main():
    #신경망 설계
    x=tf.placeholder(tf.float32,shape=[None,input_size])
    y_pred=build_autoencoder(x) #만든 데이터
    y_true=x #입력데이터가 정답데이터
    
    #loss(cost) 손실 함수 정의
    loss=tf.reduce_mean(tf.square(y_true-y_pred))    
    #optimizer 최적화 함수
    optimizer=tf.train.RMSPropOptimizer(learning_rate)    
    #train 학습 함수
    train_step=optimizer.minimize(loss)
    
    #세션을 생성, 그래프를 실행
    with tf.Session() as sess:
        #변수들의 초기값을 할당
        sess.run(tf.global_variables_initializer())
        #최적화(학습)
        for epoch in range(training_epoch):
            total_batch=int(mnist.train.num_examples/batch_size)
            for i in range(total_batch):
                batch_xs,btach_ys=mnist.train.next_batch(batch_size)
                _, current_loss=sess.run([train_step, loss],
                                         feed_dict={x:batch_xs})
            if epoch % display_step ==0:
                print('Epoch: %d,Loss: %f' %(epoch+1,current_loss))
                
        #테스트 데이터로 autoEncoding 후 Reconstruct 를 수행
        reconstructed_result=sess.run(y_pred,
                                      feed_dict={x:mnist.test.images[:examples_to_show]})
        #원본 테스트 데이터와 autoEncoding 된 데이터를 눈으로 비교
        f,a=plt.subplots(2,10,figsize=(10,2))
        for i in range(examples_to_show):
            a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
            a[1][i].imshow(np.reshape(reconstructed_result[i],(28,28)))
        f.savefig('autoEnc_mnist_image.png')
        f.show()
        plt.draw()
        plt.waitforbuttonpress()
    
if __name__=='__main__':
    main()
