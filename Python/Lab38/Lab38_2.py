# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:34:05 2018
영화 평점
@author: SDEDU
"""

from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd
from urllib.parse import urljoin 
import re
#1. 2017년 5월 1일 - 100일간 평점이 가장 좋은 1-5위 영화는?

dt_index = pd.date_range(start='20170501', end='20170806')
dt_list=dt_index.strftime('%Y%m%d').tolist()

title=['','','','','']
point=[0,0,0,0,0]
url_base='https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=cur&date='
for date in dt_list:
    url=url_base+date
    page=urlopen(url)
    soup=BeautifulSoup(page,'html.parser')
    for i in range(0,5):
        if float(soup.find_all('td','point')[i].get_text()) >point[0]:
            del title[0]
            del point[0]
            title.insert(0,(re.split(('\n|\r\n'),soup.find_all('div','tit5')[i].get_text())[1]))
            point.insert(0,float(soup.find_all('td','point')[i].get_text()))
            continue
        elif float(soup.find_all('td','point')[i].get_text()) >point[1]:
            del title[1]
            del point[1]
            title.insert(1,(re.split(('\n|\r\n'),soup.find_all('div','tit5')[i].get_text())[1]))
            point.insert(1,float(soup.find_all('td','point')[i].get_text()))
            continue
        elif float(soup.find_all('td','point')[i].get_text()) >point[2]:
            del title[2]
            del point[2]
            title.insert(2,(re.split(('\n|\r\n'),soup.find_all('div','tit5')[i].get_text())[1]))
            point.insert(2,float(soup.find_all('td','point')[i].get_text()))
            continue
        elif float(soup.find_all('td','point')[i].get_text()) >point[3]:
            del title[3]
            del point[3]
            title.insert(3,(re.split(('\n|\r\n'),soup.find_all('div','tit5')[i].get_text())[1]))
            point.insert(3,float(soup.find_all('td','point')[i].get_text()))
            continue
        elif float(soup.find_all('td','point')[i].get_text()) >point[4]:
            del title[4]
            del point[4] 
            title.insert(4,(re.split(('\n|\r\n'),soup.find_all('div','tit5')[i].get_text())[1]))
            point.insert(4,float(soup.find_all('td','point')[i].get_text()))
            continue

title
point



#2. '택시운전사'의 날짜별 평점의 변화를 그래프로 그리기
dt_index = pd.date_range(start='20170802', end='20170806')
dt_list=dt_index.strftime('%Y%m%d').tolist()
point_=[]
title_=[]
url_base='https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=cur&date='
for date in dt_list:
    url=url_base+date
    page=urlopen(url)
    soup=BeautifulSoup(page,'html.parser')
    title_.append(re.split(('\n|\r\n'),soup.find_all('div','tit5').get_text()))
    point_.append(float(soup.find_all('td','point').get_text()))

    
    
    
    
    
    title_.append(soup.find_all('a',href='/movie/bi/mi/basic.nhn?code=146469'))
    point.append(float(soup.find_all('td','point')[i].get_text()))

