# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 12:20:08 2018

@author: SDEDU
"""
import numpy as np
#import pandas as pd
#data=pd.read_csv('day.csv')

# 0: date, 1: season, 2: holiday, 3: weathersit, 4: cnt
data=np.genfromtxt('day.csv',delimiter=',',skip_header=1,usecols=[1,2,5,8,15])

#총 이용자의 수
np.sum(data[:,4],axis=0)
#이용자수가 가장 많을 때의 인원
np.max(data[:,4],axis=0)
#이용자수가 가장 적을 때의 인원
np.min(data[:,4],axis=0)
#하루 평균 이용자의 수
np.mean(data[:,4],axis=0)

##계절별 이용자 수
#봄 이용자의 총 수
np.sum(data[(data[:,1]==1),4],axis=0)
#여름 이용자의 총 수
np.sum(data[(data[:,1]==2),4],axis=0)
#가을 이용자의 총 수
np.sum(data[(data[:,1]==3),4],axis=0)
#겨울 이용자의 총 수
np.sum(data[(data[:,1]==4),4],axis=0)
#총 인원이 맞는지 확인
np.sum(data[:,4],axis=0)==np.sum(data[(data[:,1]==1),4],axis=0)+np.sum(data[(data[:,1]==2),4],axis=0)+np.sum(data[(data[:,1]==3),4],axis=0)+np.sum(data[(data[:,1]==4),4],axis=0)

##빨간날 , 파란날 이용자 수
#휴일 (holiday)
np.sum(data[(data[:,2]==0),4],axis=0)
#평일 (not holiday)
np.sum(data[(data[:,2]==1),4],axis=0)

##날씨 별 이용자의 수 
#Clear, Few clouds, Partly cloudy, Partly cloudy
np.sum(data[(data[:,3]==1),4],axis=0)
#Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
np.sum(data[(data[:,3]==2),4],axis=0)
#Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
np.sum(data[(data[:,3]==3),4],axis=0)
#Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
np.sum(data[(data[:,3]==4),4],axis=0)

#전체 인원 표준화 Normalize Temperature to 0-1 range
cnt_max=np.max(data[:,4],axis=0)
cnt_min=np.min(data[:,4],axis=0)
(data[:,4]-cnt_min)/(cnt_max-cnt_min)

#

np.sum(data[((data[:,2]==1) and (data[:,3]==1)),4],axis=0)
np.sum(data[(data[:,2]==1),4] and data[(data[:,3]==1),4],axis=0)
np.sum(data[(data[:,(2,3)]==(1,1)),4],axis=0)

data[(data[:,3]==4),4]

