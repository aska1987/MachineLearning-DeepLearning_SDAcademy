# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 10:35:36 2018

google colab 사용해서
@author: SDEDU
"""

import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets.cifar10 import load_data

#다음 배치를 읽어오기 위한 함수
#num 개수만큼 랜덤하게 이미지와 레이블을 리턴
def next_batch(num,data,labels):
    idx=np.arange(0,len(data))
    np.random.shuffle(idx)
    idx=idx[:num]
    data_shuffle=[data[i] for i in idx]
    labels_shuffle=[labels[i] for i in idx]
    
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

#CNN망을 설계 정의하는 함수
def build_CNN_classifier(x,keep_prob):
    #입력 이미지 행렬
    x_image=x
    
    #step 1 Convolution
    #Filter 5x5, 3(color) ,filter 개수: 64개
    W_conv1=tf.Variable(tf.truncated_normal(shape=[5,5,3,64],stddev=5e-2))
    b_conv1=tf.Variable(tf.truncated_normal(shape=[64],stddev=5e-2))
    c_conv1=tf.nn.conv2d(x_image,W_conv1,strides=[1,1,1,1],
                         padding='SAME')
    h_conv1=tf.nn.relu(c_conv1+b_conv1) # 32 x 32
    
    #step 1 Pooling ( 3x3 = 9 개중의 가장 큰거 뽑기)
    h_pool1=tf.nn.max_pool(h_conv1,ksize=[1,3,3,1], # 16 x 16( strides가 2x2 이고 패딩이 same이여서 1/2로 줄음)
                           strides=[1,2,2,1],padding='SAME')
    
    #step 2 convolution 64 -> 64
    W_conv2=tf.Variable(tf.truncated_normal(shape=[5,5,64,64],stddev=5e-2))
    b_conv2=tf.Variable(tf.truncated_normal(shape=[64],stddev=5e-2))
    c_conv2=tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],
                         padding='SAME')#16 x 16
    h_conv2=tf.nn.relu(c_conv2+ b_conv2)
    
    #step 2 Pooling
    h_pool2=tf.nn.max_pool(h_conv2, ksize=[1,3,3,1], #8 x 8
                           strides=[1,2,2,1],padding='SAME')
    #step 3 Convolution
    W_conv3=tf.Variable(tf.truncated_normal(shape=[3,3,64,128],stddev=5e-2))
    b_conv3=tf.Variable(tf.truncated_normal(shape=[128]))
    c_conv3=tf.nn.conv2d(h_pool2,W_conv3,strides=[1,1,1,1],
                         padding='SAME') #8 x 8
    h_conv3=tf.nn.relu(c_conv3+ b_conv3)
    
    #step 4 Convolution
    W_conv4=tf.Variable(tf.truncated_normal(shape=[3,3,128,128],stddev=5e-2))
    b_conv4=tf.Variable(tf.truncated_normal(shape=[128],stddev=5e-2))
    c_conv4=tf.nn.conv2d(h_conv3,W_conv4,strides=[1,1,1,1],
                         padding='SAME') # 8 x 8
    h_conv4=tf.nn.relu(c_conv4+ b_conv4)
    
    #step 5 Convolution
    W_conv5=tf.Variable(tf.truncated_normal(shape=[3,3,128,128],stddev=5e-2))
    b_conv5=tf.Variable(tf.truncated_normal(shape=[128],stddev=5e-2))
    c_conv5=tf.nn.conv2d(h_conv4,W_conv5,strides=[1,1,1,1],
                         padding='SAME') # 8 x 8
    h_conv5=tf.nn.relu(c_conv5+ b_conv5)
    
    #step 6 Fully Network Layer
    W_fc1=tf.Variable(tf.truncated_normal(shape=[8*8*128,384],stddev=5e-2))
    b_fc1=tf.Variable(tf.truncated_normal(shape=[384],stddev=5e-2))
    h_conv5_flat=tf.reshape(h_conv5, [-1,8*8*128])
    m_fc1=tf.matmul(h_conv5_flat,W_fc1)+b_fc1
    h_fc1=tf.nn.relu(m_fc1)
    h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
    #step 7 Hidden Network Layer
    W_fc2=tf.Variable(tf.truncated_normal(shape=[384,10],stddev=5e-2))
    b_fc2=tf.Variable(tf.truncated_normal(shape=[10],stddev=5e-2))
    logits=tf.matmul(h_fc1_drop,W_fc2)+b_fc2
    y_pred=tf.nn.softmax(logits)
    
    return y_pred, logits
#시작 함수
def main():
    #input, output, dropout을 위한 placeholder를 정의
    # None: batchsize, 32 x 32 x 3(color image)
    x=tf.placeholder(tf.float32,shape=[None,32,32,3])
    # 10개 class로 분리
    y=tf.placeholder(tf.float32,shape=[None,10])
    # 1: 100%, 0.8: 80%
    keep_prob=tf.placeholder(tf.float32)
    
    #CIFAR-10 이미지 다운 및 불러오기
    (x_train,y_train),(x_test,y_test)=load_data()
    #scalar 형태의 레이블(0~9)을 one-hot 인코딩으로 변환
    y_train_one_hot=tf.squeeze(tf.one_hot(y_train,10),axis=1)
    y_test_one_hot=tf.squeeze(tf.one_hot(y_test,10),axis=1)
    
    #CNN 그래프 생성
    #ypred는 softmax함수를 통과한값
    #logits는 softmax함수를 통과하기 전의 값
    y_pred, logits=build_CNN_classifier(x,keep_prob)
    
    #Cross Entropy 함수를 cost function으로 정의한다, logits: softmax하기 전 값
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=y,logits=logits))
    optimizer=tf.train.RMSPropOptimizer(1e-3)
    train_step=optimizer.minimize(loss)
    
    #정확도를 계산하는 연산을 추가
    correct_prediction=tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    #세션을 열고 학습 진행
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    
    loop_num=10000
    batch_size=128
    
    for i in range(loop_num):
        batch=next_batch(batch_size,x_train,sess.run(y_train_one_hot))
        
        #학습
        sess.run(train_step,feed_dict={x:batch[0], y:batch[1],keep_prob:0.8})
        
        
        # i가 10 step 마다 화면에 정확도와 loss를 출력
        if i%100 == 0:
            train_accuracy,loss_print=sess.run([accuracy,loss],
                                               feed_dict={x:batch[0],
                                                          y:batch[1],
                                                          keep_prob:1.0})
            print('Epoch: [%04d/%d], accuracy: %f, Loss: %f' %(i,loop_num,train_accuracy,loss_print))
    #학습 후 테스트 데이터(10000개) 에 대한 정확도 출력
    test_accuracy=0.0
    loop_num=10
    sample_num=1000
    for i in range(loop_num):
        test_batch=next_batch(sample_num,x_test,sess.run(y_test_one_hot))
        test_accuracy +=sess.run(accuracy, 
                                 feed_dict={x:test_batch[0],
                                            y:test_batch[1],
                                            keep_prob:1.0})
    test_accuracy /=loop_num
    print('테스트 정확도: %f' %test_accuracy)
    
    sess.close()
    
if __name__ =='__main__':
    main()