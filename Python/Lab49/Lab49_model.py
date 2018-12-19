# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 11:22:09 2018

@author: SDEDU
"""
'''
1) Q-Table : 상태,행동 점수 업데이트
학습이 잘되지만, Environment가 복잡해지면 메모리 크기가 기하급수적으로 증가, 학습시간도 기하급수적으로 증가 
-> 경우의 수가 적은 것만 사용가능

2) Q-Network: 신경망을 사용하자
1개의 신경망을 사용
1개의 신경망을 사용해서 예측값과 현재값을 추출
학습을 진행하면 예측값이 바로 변경
-> 학습 방향이 제대로 결정이 안된다
즉시 train
Local Corelation 일부 데이터에 의한 지역 상관성이 발생 
-> 학습이 잘못된 방향으로 이루어짐

3) DQN(Deep Q-Network):
    1) Go Deep: 이전 차원보다 복잡한 Environment 해결  ex) Atari, Go, StarCraft
    2) 신경망을 분리: 2개의 신경망을 사용
      main: current Action, target : predict Action
      학습시에는 main을 학습
      주기적으로 main -> target에 복사
    3) Save State: 바로 학습하지않고 일정시간동안 state를 Memory에 저장
      Memory에 저장된 데이터중에 일부를 sampling하여 이것으로 학습
      그래야 Local Corelation 의 문제를 해결하여 
      전체의 방향성과 일치하기 때문에 올바른 학습이 이루어짐
      
'''
class DQN:
    
    pass