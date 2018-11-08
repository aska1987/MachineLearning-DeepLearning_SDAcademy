# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:53:58 2018

@author: SDEDU
"""
'''
https://www.lfd.uci.edu/~gohlke/pythonlibs/ 에서 
파이썬 버전에 맞는 jpype 설치
JPype1‑0.6.3‑cp36‑cp36m‑win_amd64.whl
아나콘다 폴더에 넣은 후 프롬프트 경로를 아나콘다 폴더에 놓고 설치
(base) C:\Users\SDEDU\AppData\Local\Continuum\anaconda3>pip install JPype1-0.6.3-cp36-cp36m-win_amd64.whl

(base) C:\Users\SDEDU\AppData\Local\Continuum\anaconda3>pip install konlpy
konlpy 설치 완료

(base) C:\Users\SDEDU\AppData\Local\Continuum\anaconda3>python
stopwords & punkt 다운

데이터 전처리 및 파이썬 자연어 처리 라이브러리 정리
http://aileen93.tistory.com/128

'''

from konlpy.tag import Kkma
kkma=Kkma()
kkma.sentences('한국어 분석을 시작합니다 재미있어요--')
kkma.nouns('한국어 분석을 시작합니다 재미있어요--') 
kkma.pos('한국어 분석을 시작합니다 재미있어요--') #형태소 분석

from konlpy.tag import Hannanum
hannanum=Hannanum()
hannanum.nouns('한국어 분석을 시작합니다 재미있어요--')
hannanum.pos('한국어 분석을 시작합니다 재미있어요--')

from konlpy.tag import Twitter
t=Twitter()
t.nouns('한국어 분석을 시작합니다 재미있어요--')
t.morphs('한국어 분석을 시작합니다 재미있어요--')
t.pos('한국어 분석을 시작합니다 재미있어요--')

from wordcloud import WordCloud, STOPWORDS
import numpy as np
from PIL import Image
text=open('DataScience-master\\data\\09. alice.txt').read()
alice_mask=np.array(Image.open('DataScience-master\\data\\09. alice_mask.png'))
stopwords=set(STOPWORDS)
stopwords.add('said')

import matplotlib.pyplot as plt
import platform

path='c:\Windows\Fonts\malgun.ttf'
from matplotlib import font_manager, rc
if platform.system()=='Darwin':
    rc('font',family='AppleGothic')
elif platform.system()=='Windows':
    font_name=font_manager.FontProperties(fname=path).get_name()
    rc('font',family=font_name)
else:
    print('Unknown system')

plt.figure(figsize=(8,8))
#plt.imshow(alice_mask,cmap=plt.cm.gray, interpolation='nearest')
plt.imshow(alice_mask,cmap=plt.cm.gray, interpolation='bilinear')
plt.axis('off')

wc=WordCloud(background_color='white',max_words=2000,mask=alice_mask,
             stopwords=stopwords)
wc=wc.generate(text)
wc.words_

plt.figure(figsize=(12,12))
plt.imshow(wc,interpolation='bilinear')
ptl.axis('off')
plt.show()

import nltk
from konlpy.corpus import kobill
files_ko=kobill.fileids()
doc_ko=kobill.open('1809890.txt').read()
doc_ko

from konlpy.tag import Twitter
t=Twitter()
tokens_ko=t.nouns(doc_ko)
tokens_ko

ko=nltk.Text(tokens_ko,name='대한민국 국회 의안 제 1909890호')
print(len(ko.tokens))
print(len(set(ko.tokens))) #unique tokens
ko.vocab() #frequency


plt.figure(figsize=(12,6))
ko.plot(50)

stop_words=['의','자','에','안','번','호','을','이','다','만','로','가','를']
ko=[each_word for each_word in ko
    if each_word not in stop_words]
ko

ko.count('초등학교')
plt.figure(figsize=(12,6))
ko.dispersion_plot(['육아휴직','초등학교','공무원'])

ko.concordance('초등학교')

ko.collocations()

data=ko.vocab().most_common(150)


wordcloud = WordCloud(font_path='c:\\Windows\\Fonts\\malgun.ttf',
                      relative_scaling = 0.2,
                      background_color='white',
                      ).generate_from_frequencies(dict(data))

plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.axis('off')


#1. 영문 키워드 분석해서 워드클라우드 만들기
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from konlpy.corpus import kobill

calumn1=pd.read_excel('칼럼(2012-2014).XLS',sheet_name='2014년')
calumn2=pd.read_excel('칼럼(2012-2014).XLS',sheet_name='2013년')
calumn3=pd.read_excel('칼럼(2012-2014).XLS',sheet_name='2012년')

#인덱스번호 안겹치게 순차적으로 하기 위해서
calumn4=pd.concat([calumn3,calumn1,calumn2],axis=0,ignore_index=True) 

calumn4=calumn4.replace(' ',np.nan) #빈칸 nan로 처리
cal.isnull().sum()
df=calumn4['영문키워드']
df=df.dropna()
df.index=pd.RangeIndex(len(df.index)) #인덱스 0부터 순서대로 설정

df_list=[]
for i in range(0,len(df)):
    df_list.append(df[i].split(';'))
df_list
df_str=''
for i in range(0,len(df_list)):
    df_str+=" ".join(df_list[i])


wordcloud=WordCloud().generate(df_str)
wordcloud.words_

plt.figure(figsize=(12,12))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')


#2. 한글 키워드 분석해서 워드클라우드 작성

hangle=calumn4['국문키워드']
hangle.shape
hangle=hangle.dropna()
hangle.index=pd.RangeIndex(len(hangle.index))
hangle_list=[]
for i in range(0,len(hangle)):
    hangle_list.append(hangle[i].split(';'))
hangle_list
hangle_str=''
for i in range(0,len(hangle_list)):
    hangle_str+=" ".join(hangle_list[i])
    
hangle_str

from konlpy.tag import Twitter
t=Twitter()
tokens_ko=t.nouns(hangle_str)
tokens_ko
ko=nltk.Text(tokens_ko,name='한글분석')
plt.figure(figsize=(12,6))
ko.plot(50)

data=ko.vocab().most_common(150)

wordcloud = WordCloud(font_path='c:\\Windows\\Fonts\\malgun.ttf',
                      relative_scaling = 0.2,
                      background_color='white',
                      ).generate_from_frequencies(dict(data))

plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.axis('off')

import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x>0,dtype=np.int)

x=np.arange(-5.0,5.0,0.1)
y=step_function(x)
plt.plot(x,y) #직선
plt.ylim(-0.1,1.1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

x=np.array([-1.0,1.0,2.0])
sigmoid(x)

x=np.arange(-5.0,5.0,0.1)
y=sigmoid(x)
plt.plot(x,y) #곡선으로
plt.ylim(-0.1,1.1)

x=np.arange(-5.0,5.0,0.1)
y1=sigmoid(x)
y2=step_function(x)

#계단형 시그모이드 비교(선형 비선형) 
plt.plot(x,y1,label='sigmoid')
plt.plot(x,y2,linestyle='--',label='step_function')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-0.1,1.1)

A=np.array([[1,2],[3,4]])
print(A.shape)
B=np.array([[5,6],[7,8]])
print(B.shape)
np.dot(A,B) #곱 (내적)

x=np.array([1,2])
print(x.shape)
w=np.array([[1,3,5],[2,4,6]])
ww.shape
y=np.dot(x,w)
y
