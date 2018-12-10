# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:52:49 2018

@author: SDEDU
"""
'''
비행기가 어떤 조건에서 이륙할 때 얼마만큼의 이륙거리가 필요한지 예측
가상의 비행기 B777-200 이륙데이터사용
- 이륙속도 : 290km/h
- 최대비행기 무게 : 300ton
- 필요한 활주거리 : 2000m
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#데이터 셔플
def shuffle_data(x_data,y_data):
    temp_index=np.arange(len(x_data))
    
    np.random.shuffle(temp_index)
    x_temp=np.zeros(x_data.shape)
    y_temp=np.zeros(y_data.shape)
    x_temp=x_data[temp_index]
    y_temp=y_data[temp_index]
    return x_temp,y_temp


'''
데이터 졍규화
-> 이륙속도,무게 와 같이 서로다른단위를 일정한 비율로 맞춰주는 작업
 0 ~ 1 사이 값으로 정규화
'''
def minmax_normalize(x):
    xmax, xmin=x.max(), x.min()
    norm=(x-xmin)/(xmax-xmin)
    return norm

#정규화 후 realx에 해당하는 정규값 리턴
def minmax_get_norm(realx,arrx):
    xmax,xmin=arrx.max(), arrx.min()
    normx=(realx - xmin)/ (xmax - xmin)
    return normx

#훈련을 끝내고 원래 단위로 변경(역정규화)
def minmax_get_denorm(normx,arrx):
    xmax,xmin=arrx.max(),arrx.min()
    realx=normx * (xmax-xmin) + xmin
    return realx

def main():
    traincsvdata=np.loadtxt('airplane/trainset.csv',unpack=True,
                            delimiter=',',skiprows=1)
    num_points=len(traincsvdata[0])
    print('points: ' , num_points)
    
if __name__=='__main__':
    main()