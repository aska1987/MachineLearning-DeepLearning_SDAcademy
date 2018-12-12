# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 09:33:34 2018
Tensorboard
@author: SDEDU
"""
'''
개인적으로 http://jaynewho.com/post/8 참조
'''

import tensorflow as tf
a=tf.constant(3.0,name='a')
b=tf.constant(4.0,name='b')
c=tf.add(a,b,name='addnode')

#step 1. scalar 기록한 node 선택
tf.summary.scalar('add_result_c',c)

#step 2. summary 통합
mergeall=tf.summary.merge_all()

with  tf.Session() as sess:
    print('c= ',sess.run(c))
    
    #step 3. writer 생성
    writer=tf.summary.FileWriter('tmp/tensorboard/addnum')
    writer.add_graph(sess.graph)
    
    #step 4. summary 정보 로그 추가
    summary=sess.run(mergeall)
    writer.add_summary(summary)
    
''' anaconda prompt
cd C:\Users\SDEDU\Documents\GitHub\MachineLearning-DeepLearning_SDAcademy\Python\Lab45
tensorboard --logdir=./tmp/tensorboard/addnum
크롬에 http://localhost:6006/#scalars
'''

a=tf.placeholder(tf.float32,name='a')
b=tf.constant(5.0,name='b')
c=tf.add(a,b,name='add_op')
#step 1. scalars 기록할 node 선택
tf.summary.scalar('add_result_c',c)

#step 2: summary 통합
merged=tf.summary.merge_all()


with tf.Session() as sess:
    #step 3: writer 생성
    writer=tf.summary.FileWriter('tmp/tensorboard/fori100',sess.graph)
    for step in range(100):
        #step 4: 그래프 실행, summary 정보 로그 추가
        print(sess.run(c,feed_dict={a:step *1.0}))
        summary=sess.run(merged,feed_dict={a:step*1.0})
        writer.add_summary(summary,step)
        
        

a=tf.constant(2.0,name='a')
b=tf.constant(3.0,name='b')
c=tf.constant(4.0,name='c')


#name_scope는 영역을 의미 (Graph에서는 그룹박스로 표시됨)
with tf.name_scope('large_op'):
    with tf.name_scope('mini_op'):
        addop=tf.add(a,b,name='add_op')
    mulop=tf.multiply(addop,c,name='mul_op')

#step 1: scalars 기록할 node 선택
tf.summary.scalar('mul_result',mulop)

#step 2: summary 통합
merged=tf.summary.merge_all()

with tf.Session() as sess:
    #step 3: writer 생성
    writer=tf.summary.FileWriter('tmp/tensorboard/group',
                                 sess.graph)
    #step 4: 그래프 실행, summary 정보 로그 추가
    summary=sess.run(merged)
    writer.add_summary(summary)
print('Done')


## 변수 Save ######################################### 
b1=tf.Variable(2.0,name='bias')

#Saver 생성(학습데이터를 저장하기 위한 객체)
saver=tf.train.Saver()

sess=tf.Session()
sess.run(tf.global_variables_initializer())
print('save test bias: ',sess.run(b1))
#변수 저장
save_path=saver.save(sess,'saver_bias/bias.ckpt')
print('Model saved in file: %s' %save_path)

sess.close()


## 변수 Restore ####################################
b1=tf.Variable(0.0, name='bias')
saver=tf.train.Saver()
sess=tf.Session()

#읽어들일때는 ses.run(tf.global_variables_initializer()) 초기화 하지않음
ckpt=tf.train.get_checkpoint_state('saver_bias')
if tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
    
    print('variable is restored')
print('bias: ', sess.run(b1))


