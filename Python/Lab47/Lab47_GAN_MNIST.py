# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 10:39:18 2018

@author: SDEDU
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('./mnist/data/',one_hot=True)

total_epoch=100 #총 반복횟수
batch_size=100 #1회 학습 분량
learning_rate=0.0002
n_input=784 
n_hidden=256
n_noise=128 #임의의 데이터 -> 학습 ->MNIST

#신경망 모델 구성
x=tf.placeholder(tf.float32,[None,n_input]) #mnist real데이터  -  Real
z=tf.placeholder(tf.float32,[None,n_noise]) #임의의 렌덤 노이즈 -  Fake

#Generative - Fake MNIST 제작
G_W1=tf.Variable(tf.random_normal([n_noise,n_hidden],stddev=0.01))
G_b1=tf.Variable(tf.zeros([n_hidden]))

G_W2=tf.Variable(tf.random_normal([n_hidden,n_input],stddev=0.01))
G_b2=tf.Variable(tf.zeros([n_input]))

#Discriminator - Real or Fake 구분
#1이면 Real, 0이면 Fake
D_W1=tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1=tf.Variable(tf.zeros([n_hidden]))

D_W2=tf.Variable(tf.random_normal([n_hidden,1],stddev=0.01))
D_b2=tf.Variable(tf.zeros([1]))


#Fake MNIST 생성
#128 -> 256 -> 784
def generator(noise_z):
    hidden=tf.nn.relu(tf.matmul(noise_z,G_W1)+G_b1)
    output=tf.nn.sigmoid(tf.matmul(hidden,G_W2)+G_b2)
    return output

#noise 값을 생성하는 함수
def get_noise(batch_size,n_noise):
    return np.random.normal(size=(batch_size,n_noise))

#진짜면 1, 가짜면 0 으로 반환하는 구분 함수
#784 -> 256 -> 1
def discriminator(inputs):
    hidden=tf.nn.relu(tf.matmul(inputs,D_W1)+D_b1)
    output=tf.nn.sigmoid(tf.matmul(hidden,D_W2)+D_b2)
    return output

G=generator(z) #Make Fake MNIST 
D_gene=discriminator(G) # input : fake MNIST --> 가짜로 판정 내리도록 학습 output : 0
D_real=discriminator(x) # input : real_MNIST --> 진짜로 판정 내리도록 학습 output : 1

#loss(cost) 함수
#tf.log(1) =0
#tf.log(0) = - infinite
#loss_D = 0으로 하려면 D_real=1 이 되도록, D_gene=0이 되도록 학습
loss_D=tf.reduce_mean(tf.log(D_real)+tf.log(1-D_gene)) #이 과정에서 D_gene값을 0에 가깝게 하려함
#loss_G = 1이 되도록 해야 진짜 같은 Fake가 된다
loss_G=tf.reduce_mean(tf.log(D_gene)) #이 과정에서 D_gene값을 1에 가깝게 하려함 

D_var_list=[D_W1,D_b1,D_W2,D_b2] #구분자와 연결된 학습 변수들 리스트
G_var_list=[G_W1,G_b1,G_W2,G_b2] #생성자와 연결된 학습 변수들 리스트

#train 함수(optimizer함수 포함)
#음수값 -> 결과값이 0으로 나오도록 학습을 시켜야 함
#maximize 함수가 없다보니 minimize 를 이용하여 - log(x) 로 하여 maximize 화 한다 -> 0에가깝게
train_D=tf.train.AdamOptimizer(learning_rate).minimize(-loss_D,
                              var_list=D_var_list)
train_G=tf.train.AdamOptimizer(learning_rate).minimize(-loss_D,
                              var_list=G_var_list)


#GAN 신경망 모델 학습
sess=tf.Session()
sess.run(tf.global_variables_initializer())

total_batch=int(mnist.train.num_examples/batch_size)
loss_val_D,loss_val_G = 0,0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs,batch_ys =mnist.train.next_batch(batch_size) #Real MNIST
        noise= get_noise(batch_size,n_noise) #noise
        
        #loss_val_D 가 점점 0에 가까워 지는 방향으로 학습
        #real 데이터와 fake 데이터를 잘 구분하는 방향으로 학습
        _,loss_val_D=sess.run([train_D,loss_D],
                              feed_dict={x:batch_xs, z:noise})
        #loss_val_G 가 점점 0에 가까워 지는 방향으로 학습
        #fake 데이터를 real 처럼 만드는 방향으로 학습
        _,loss_val_G=sess.run([train_G,loss_G],
                              feed_dict={z:noise})
    print('epoch: ', '%04d' %epoch,
          'D loss: {:.4}'.format(loss_val_D),
          'G loss: {:.4}'.format(loss_val_G))
    #확인용 가짜 이미지 생성
    #10번 epoch 마다 점점 이미지가 개선되는지 알아버려 이미지 생성
    if epoch ==0 or (epoch+1) %10 ==0:
        sample_size=10
        noise=get_noise(sample_size, n_noise) #noise
        samples=sess.run(G,feed_dict={z:noise}) #noise - Fake MNIST
        fig, ax=plt.subplots(1,sample_size, figsize=(sample_size,1))
        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i],(28,28)))
        plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)),
                    bbox_inches='tight')
        plt.close(fig)
print('최적화 완료')
sess.close()