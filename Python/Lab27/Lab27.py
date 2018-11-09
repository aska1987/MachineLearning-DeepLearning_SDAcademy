# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 09:35:35 2018

@author: SDEDU
"""

#수치 미분(=아주 작은 차분으로 미분을 구하는 것)
def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)
#y=0.01x^2 + 0.1x
def function_1(x):
    return 0.01*x**2 + 0.1*x
#그래프로 그리기
import numpy as np
import matplotlib.pylab as plt
x=np.arange(0.0,20.0,0.1)
y=function_1(x)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x,y)

numerical_diff(function_1,5) # 0.1999999999990898
numerical_diff(function_1,10) # 0.2999999999986347
'''
y=0.01x^2 + 0.1x 의 미분은
df(x)/dx=0.02x +0.1 이 된다
여기서 x가 5,10일때 진정한 미분은 0.2, 0.3이 됨
앞의 수치 미분과 결과를 비교하면 오차가 매우 작다.
'''
#편미분 : 변수가 여럿인 함수에 대한 미분. 어느 변수에 대한 미분이냐를 구별해야함
#f(x_0,x_1)=x_0 ^2 + x_1 ^2
def function_2(x):
    return x[0]**2 + x[1]**2

#그래프
from mpl_toolkits.mplot3d import Axes3D
X=np.arange(-3,3,0.25)
Y=np.arange(-3,3,0.25)
XX,YY=np.meshgrid(X,Y)
ZZ=XX**2 + YY**2
fig=plt.figure()
ax=Axes3D(fig)
ax.plot_surface(XX,YY,ZZ,rstride=1,cstride=1,cmap='hot')

#x_0=3, x_1=4 일떄 x0에 대한 편미분을 구하라
def function_tmp1(x0):
    return x0*x0 + 4.0**2.0
numerical_diff(function_tmp1,3.0)
#x1에 대한 편미분
def function_tmp2(x1):
    return 3.0**2.0 +x1*x1
numerical_diff(function_tmp2,4.0)

#기울기 : 모든 변수의 편미분을 벡터로 정리한 것
def _numerical_gradient_no_batch(f,x):
    h=1e-4
    grad=np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val=x[idx]
        
        #f(x+h)
        x[idx]=float(tmp_val)+h
        fxh1=f(x)
        
        #f(x-h)
        x[idx]=tmp_val-h
        fxh2=f(x)
        
        grad[idx]=(fxh1-fxh2) /(2*h)
        x[idx]=tmp_val
    return grad

#(3,4) (0,2) (3,0) 에서의 기울기  
_numerical_gradient_no_batch(function_2,np.array([3.0,4.0]))
_numerical_gradient_no_batch(function_2,np.array([0.0,2.0]))
_numerical_gradient_no_batch(function_2,np.array([3.0,0.0]))
'''
경사법(경사 하강법) : 기울기를 잘 이용해 함수의 최소값을 찾으려는 방법
최적의 매개변수을 학습시에 찾음. 최적은 손실함수가 최가값이 될 때 매개변수값
함수가 극소값,최소값,안장점이 되는 장소에서는 기울기가 0
극소값: 한정된 범위에서의 최소값인 점
안장점: 어느 방향에서 보면 극대값 다른 방향에서 보면 극소값이 되는 점

학습률 너무 크면 큰 값으로 발산함
학습률이 너무 작으면 거의 갱신되지 않은 채 끝남
적절한 값을 찾아내는 과정이 필요
'''

def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad
#경사 하강법
# f: 최적화 하려는 함수, lr : learnuing rate(학습률), step_num: 경사하강법 반복횟수
def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x=init_x
    for i in range(step_num):
        grad=numerical_gradient(f,x) #함수 기울기
        x-=lr*grad #기울기 * 학습률 
        return x
    
init_x=np.array([-3.0,4.0])
gradient_descent(function_2,init_x=init_x, lr=0.1,step_num=100)

#그래프
def gradient_descent(f, init_x,lr=0.01, step_num=100):
    x=init_x
    x_history=[]
    
    for i in range(step_num):
        x_history.append(x.copy())
        grad=numerical_gradient(f,x)
        x-=lr*grad
    return x,np.array(x_history)
init_x=np.array([-3.0,4.0])
lr=0.1
step_num=20
x,x_history=gradient_descent(function_2,init_x,lr=lr,step_num=step_num)

plt.plot([-5,5],[0,0],'--b')
plt.plot([0,0],[-5,5],'--b')
plt.plot(x_history[:,0],x_history[:,1],'o')

plt.xlim(-3.5,3.5)
plt.ylim(-4.5,4.5)
plt.xlabel('X0')
plt.ylabel('X1')

#
import pandas as pd
import numpy as np
data=pd.read_csv('1109_data//02. crime_in_Seoul.csv',thousands=',',
                 encoding='euc-kr')

import os
import sys
import urllib.request
import googlemaps
client_id='_M6QMrUwSzIQrzLnWOMj'
client_secret='MdCHqJzc1D'
encText = urllib.parse.quote("서울중부경찰서")
url = "https://openapi.naver.com/v1/map/geocode?query=" + encText # json 결과
# url = "https://openapi.naver.com/v1/map/geocode.xml?query=" + encText # xml 결과
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)

gmaps_key='AIzaSyCbYLe2eKlAkxNddQ59hJY20hl62TpmEZ8'
gmaps=googlemaps.Client(key=gmaps_key)    
gmaps.geocode('서울중부경찰서', language='ko')

station_name = []

for name in data['관서명']:
    station_name.append('서울' + str(name[:-1]) + '경찰서')

station_name

station_address=[]
station_lat=[]
station_lng=[]
for name in station_name:
    tmp=gmaps.geocode(name,language='ko')
    station_address.append(tmp[0].get('formatted_address'))
    
    tmp_loc=tmp[0].get('geometry')
    station_lat.append(tmp_loc['location']['lat'])
    station_lng.append(tmp_loc['location']['lng'])
    print(name+'-->'+tmp[0].get('formatted_address'))
    

gu_name = []

for name in station_address:
    tmp = name.split()
    
    tmp_gu = [gu for gu in tmp if gu[-1] == '구'][0]
    
    gu_name.append(tmp_gu)
    
data['구별'] = gu_name
data.head()

#데이터 분석
#1. 제목의 키워드 분석
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud
from konlpy.tag import Twitter
from collections import Counter
from matplotlib import font_manager,rc
from fbprophet import Prophet
data1=pd.read_excel('사이트방문자수(2012-2014).xls',sheet_name='2014년')
data2=pd.read_excel('사이트방문자수(2012-2014).xls',sheet_name='2013년')
data3=pd.read_excel('사이트방문자수(2012-2014).xls',sheet_name='2012년')
data=pd.concat([data1,data2,data3],axis=0,ignore_index=True)


data_title=data['제목']
title_str=''
for i in range(0,len(data_title)):
    title_str+="".join(data_title[i])
title_str

spliter=Twitter()
tokens_ko=spliter.nouns(title_str)
tokens_ko
ko=nltk.Text(tokens_ko,name='한글분석')

#한글 깨질 경우
font_name=font_manager.FontProperties(fname='c:/Windows/Fonts/malgun.ttf').get_name()
rc('font',family=font_name)

plt.figure(figsize=(12,6))
ko.plot(50)
import numpy as np
y=np.arange(0,6,1)


title_counts=ko.vocab().most_common(150)

for i in range(0,len(title_counts)):
    title_counts[i].split(',')
title_counts[0].split(',')



wordcloud = WordCloud(font_path='c:\\Windows\\Fonts\\malgun.ttf',
                      relative_scaling = 0.2,
                      background_color='white',
                      ).generate_from_frequencies(dict(title_counts))

plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.axis('off')
#2. 6개월 후 방문객 수 예측
df=pd.DataFrame({'ds':data['날짜'],'y':data['방문']})
df.head()

m=Prophet(yearly_seasonality=True)
m.fit(df)
future=m.make_future_dataframe(periods=120)
future.tail()
forecast=m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
m.plot(forecast)
forecast.tail(1)
m.plot_components(forecast)

#3. 향후 어떤 기사를 실었으면 좋을지?
