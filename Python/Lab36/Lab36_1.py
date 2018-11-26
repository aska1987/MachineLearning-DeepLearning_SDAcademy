# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:43:29 2018
주유소 데이터, 인구 데이터 
@author: SDEDU
"""

import pandas as pd
from glob import glob
glob('data/*.xls')

stations_files=glob('data/*.xls')
stations_files

tmp_raw=[]
for file_name in stations_files:
    tmp=pd.read_excel(file_name,header=2)
    tmp_raw.append(tmp)
station_raw=pd.concat(tmp_raw)
station_raw.info()

stations=pd.DataFrame({'Oil_store':station_raw['상호'],
                       '주소':station_raw['주소'],
                       '가격':station_raw['휘발유'],
                       '셀프':station_raw['셀프여부'],
                       '상표':station_raw['상표']})
stations
stations['구']=[eachAddress.split()[1] for eachAddress in stations['주소']]
stations['구'].unique()

stations[stations['구']=='서울특별시']
stations.loc[stations['구']=='서울특별시','구']='성동구'
stations['구'].unique()

stations[stations['구']=='특별시']
stations.loc[stations['구']=='특별시','구']='도봉구'
stations['구'].unique()

stations=stations[stations['가격'] !='-']

stations['가격']=[float(value) for value in stations['가격']]
stations.reset_index(inplace=True)
del stations['index']

stations.info()


## 셀프 주유소는 저렴한지 BOXPLOT 으로 확인하기 #########################
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import numpy as np
path='c:/Windows/Fonts/malgun.ttf'
from matplotlib import font_manager,rc
if platform.system()=='Darwin':
    rc('font',family='AppleGothic')
elif platform.system()=='Windows':
    font_name=font_manager.FontProperties(fname=path).get_name()
    rc('font',family=font_name)
else:
    print('unknown')

stations.boxplot(column='가격',by='셀프',figsize=(12,8))

plt.figure(figsize=(12,8))
sns.boxplot(x='상표',y='가격',hue='셀프',data=stations,palette='Set3')

plt.figure(figsize=(12,8))
sns.boxplot(x='상표',y='가격',data=stations,palette='Set3')
sns.swarmplot(x='상표',y='가격',data=stations,color='.6')
#서울에서 제일 높고 낮은 가격의 주유소는?
stations.sort_values('가격').head(1)
stations.sort_values('가격').tail(1)
#구별 주유소 가격
gu_data=pd.pivot_table(stations,index=['구'],values=['가격'],aggfunc=np.mean)
gu_data

## 우리나라 인구가 소멸될 도시 맞추기 ##############################
#젋은 여성 인구가 노인 인구의 절반에 미달할 경우 '소멸 위험 지역' 으로 분류'

data=pd.read_excel('population_data.xlsx',header=1)
data.isnull().sum()
df=data.fillna(method='ffill')
df.columns

young_data=df['20 - 24세']+df['25 - 29세']+df['30 - 34세']+df['35 - 39세']
old_data=df['65 - 69세']+df['70 - 74세']+df['75 - 79세']+df['80 - 84세']+df['85 - 89세']+df['90 - 94세']+df['95 - 99세']+df['100+']
old_data.shape
young_data.shape

df['young']=young_data
df['old']=old_data

a=df[df['항목']=='여자인구수 (명)']
b=df[df['항목']=='총인구수 (명)']

a.reset_index(inplace=True)
del a['index']
b.reset_index(inplace=True)
del b['index']
a.shape
b.shape

a['노인 비율']=(a['young'])/(b['old'])
a.loc[a['노인 비율']>=0.5,'소멸위험지역여부']=0 #no
a.loc[a['노인 비율']<0.5,'소멸위험지역여부']=1 #yes
temp=a[a['소멸위험지역여부']==1]
temp.shape
temp=temp[temp['행정구역(동읍면)별(2)']!='소계']
temp.shape

