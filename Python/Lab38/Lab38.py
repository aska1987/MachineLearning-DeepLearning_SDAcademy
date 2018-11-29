# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 10:57:58 2018

@author: SDEDU
"""

from bs4 import BeautifulSoup
page=open("example.html",'r').read()
soup=BeautifulSoup(page,'html.parser')
print(soup.prettify()) #html 페이지의 내용 전체 보기

list(soup.children) #자식만 찾기

html=list(soup.children)[1]
html
list(html.children)

body=list(html.children)[3]
body

soup.body #body만 찾기

list(body.children)

len(list(body.children))

soup.find_all('p') #모든 p태그 찾기
soup.find('p') #첫 p태크만 찾기

soup.find_all('p',class_='outer-text') #class가 outer-text 인 것 찾기
soup.find_all(id='first') #id가 first 인것 찾기
soup.head #헤드만 출력

soup.head.next_sibling #head 다음에 줄바꿈 문자
soup.head.next_sibling.next_sibling #head 다음 다음 body

body.p #body의 첫번째 p
body.p.next_sibling.next_sibling #body의 두번쨰 p

for each_tag in soup.find_all('p'):
    print(each_tag.get_text())
    
body.get_text() #태그가 있던 자리는 줄바꿈(\n)이 표시되고 전체 텍스트를 보여줌
links=soup.find_all('a')
links

for each in links:
    href=each['href']
    text=each.string
    print(text+ '->' + href)


## 크롬 개발자 도구를 이용해서 원하는 태그 찾기
from urllib.request import urlopen
url='http://info.finance.naver.com/marketindex/'
page=urlopen(url)
soup=BeautifulSoup(page,'html.parser')
soup.prettify()
#---> 페이지에 있는 태그 다 가져옴

soup.find_all('span','value')[0].string #환율 정보만

url_base='http://www.chicagomag.com'
url_sub='/Chicago-Magazine/November-2012/Best-Sandwiches-Chicago/'
url=url_base+url_sub

html=urlopen(url)
soup=BeautifulSoup(html,'html.parser')
soup.find_all('div','sammyListing')
len(soup.find_all('div','sammyListing'))
soup.find_all('div','sammyListing')[0]#첫번째 맛집
tmp_one=soup.find_all('div','sammy')[0]#첫번째 맛집
type(tmp_one)
tmp_one.find(class_='sammyRank')
tmp_one.find(class_='sammyRank').get_text()

tmp_one.find('a')['href']

import re #정규식
tmp_string=tmp_one.find(class_='sammyListing').get_text()
re.split(('\n|\r\n'),tmp_string) #분리
print(re.split(('\n|\r\n'),tmp_string[0])) #메뉴이름
tmp_string[1] #가계이름

from urllib.parse import urljoin #절대경로로 잡힌 url은 그대로 두고 상대경로는 절대경로로 변경
rank=[] #순위
main_menu=[] #메인메뉴이름
cafe_name=[] #카페 이름
url_add=[] #접근 주소
list_soup=soup.find_all('div','sammy')
for item in list_soup:
    rank.append(item.find(class_='sammyRank').get_text())
    
    tmp_string=item.find(class_='sammyListing').get_text()
    main_menu.append(re.split(('\n|\r\n'),tmp_string)[0])
    cafe_name.append(re.split(('\n|\r\n'),tmp_string)[1])
    
    url_add.append(urljoin(url_base,item.find('a')['href']))

import pandas as pd
data={'Rank':rank,'Menu':main_menu,'Cafe':cafe_name,'URL':url_add}
df=pd.DataFrame(data)

