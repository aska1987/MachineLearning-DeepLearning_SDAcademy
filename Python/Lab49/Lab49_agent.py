# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 11:21:19 2018

@author: SDEDU
"""

import tensorflow as tf
import numpy as np
import random
import time

from Lab49_game import Game
from Lab49_model import DQN

MAX_EPISODE=10000 # 최대 학습수
TARGET_UPDATE_INTERVAL=1000 # main->target NN으로 학습결과 update 주기
TRAIN_INTERVAL=4 # 몇 프레임마다 학습 진행(state를 memory에 저장)
OBSERVE=100 # 100번 반복 이후에 action를 받아옴
dNUM_ACTION=3 # 0좌, 1유지, 2우 action의 개수
SCREEN_WIDTH=6 #일반적으로 pixel개수를 주지만 이곳은 빠른
SCREEN_HEIGHT=10 # 학습을 위해서 블록 크기정도로 줌


def train(): #강화학습
    print('Training.....')
    sess=tf.Session()
    
    game=Game(SCREEN_WIDTH,SCREEN_HEIGHT,show_game=False)
    brain=DQN(sess,SCREEN_WIDTH,SCREEN_HEIGHT,NUM_ACTION)
    
    rewards=tf.placeholder(tf.float32,[None])
    tf.summary.scalar('avg.reward/ep.',tf.reduce_mean(rewards))
    
    saver=tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    writer=tf.summary.FileWriter('logs',sess.graph)
    summary_merged=tf.summary.merge_all()
    
    brain.update_target_network() #main 학습변수 ->target 학습변수 복사
    
    epsilon=1.0 #처음에는 임의대로, 나중에는 섞어서
    time_step=0 #프레임 횟수
    total_reward_list=[] #진행중에 reward들의 모음
    
    #게임을 시작
    for episode in range(MAX_EPISODE):
        terminal=False #게임이 끝낫는지 여부
        total_reward=0 #게임 전체 보상점수
        
        #MDP: Markov Decision Process
        # 1) Get State 2) Decision Action 3)Get Reward
        # 1) 상태 획득(Get State)
        state=game.reset() #게임 초기화
        brain.init_state(state) #신경망 초기화
        '''
        초반에는 학습이 전혀 안되어 있으므로 임의로 행동을 결정
        100회 게임 이상부터 epsilon 값을 줄여서 가끔 신경망으로부터
        action을  받아오다가 점점 거듭할 수록 신경망으로 action을 받아오는 회수 증가
        '''
        while not terminal: #게임이 끝나지 않으면
            # 2) 동작 결정(Decision Action)
            if np.random.rand() < epsilon:
                action=random.randrange(NUM_ACTION)
            else:
                action=brain.get_action()
            
            if episode>OBSERVE:
                epsilon -=1/1000
            # 3) 보상(상/벌점)(Get Reward)
            # 게임을 action 으로 진행(game.step(action))해서 
            state, reward, terminal=game.step(action)
            total_reward += reward #상벌점을 누적
            
            # 현재 상태를 brain에 기억
            # Memory에 계속 누적
            brain.remember(state,action,reward,terminal)
            
            # 100번이상 게임진행되고 이후로 4번마다 1번씩 학습을 진행
            if time_step > OBSERVE and time_step % TRAIN_INTERVAL ==0:
                brain.train()
                
            #기본신경망(main)이 학습이 되었다면 학습결과를 타겟신경망에 복사
            if time_step % TARGET_UPDATE_INTERVAL ==0:
                brain.update_target_network()
            time_step +=1
        print('게임횟수: %d 점수: %d' %(episode+1, total_reward))
        total_reward_list.append(total_reward)
        
        if episode %10==0: #10회당 1번씩 그래프 로그 저장
            summary=sess.run(summary_merged,
                             feed_dict={rewards:total_reward_list})
            writer.add_summary(summary,time_step)
            total_reward_list=[]
        if episode %100==0 : #게임 100회당 1번씩 학습 값 저장
            saver.save(sess,'model/dqn.ckpt',global_step=time_step)
def replay(): #알아서 플레이
    print(')

def menu()_: #강화학습 or 플레이
    print('0. train')
    print('1. replay')
    sel =int(input('select Number:'))
    return sel

def main(): #시작
    sel=menu()
    if sel==0:
        train()
    elif sel==1:
        replay()
    else:
        print('Invalid Select Number')
        

if __name__=='__main__':
    main()