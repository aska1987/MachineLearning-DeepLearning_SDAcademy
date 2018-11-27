# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:45:34 2018
cctv 현황 분석하여 그래프로 나타내기
@author: SDEDU
"""

#1. cctv 현황 - 수평바
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('CCTV_re.csv')
data
data.isnull().sum()


from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name) 
# --> 한글 꺠짐 방지

plt.barh(data['구별'],data['CCTV비율'])

#2. 인구대비 cctv 비율계산해서 정렬하여 나타내기
data.sort_values(['CCTV비율'])


#3. scatter 함수를 사용해서 s = 50 마커 크기로 나타내기
plt.scatter(data['구별'],data['소계'],s=50)

#4. 적당한 직선을 그리기(인구 300000일 때는 CCTV는 110정도 한다는 개념)
import seaborn as sns
plt.plot([300000,110])
sns.lmplot(x='인구수',y='소계',data=data)
sns.lmplot(x='인구수',y='소계',data=data,order=1)
sns.lmplot(x='인구수',y='소계',data=data,order=2)
sns.lmplot(x='인구수',y='소계',data=data,order=3)
sns.lmplot(x='인구수',y='소계',data=data,order=4)
sns.lmplot(x='인구수',y='소계',data=data,order=5)

#5. 4. 를 기준으로 텍스트와 color map로 입히기 
sns.lmplot(x='인구수',y='소계',data=data,palette='hls')
plt.xlabel('인구수')
plt.ylabel('CCTV수')
