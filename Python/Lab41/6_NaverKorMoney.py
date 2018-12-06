#exchangeList > li.on > a.head.usd > div > span.value

from bs4 import BeautifulSoup
import urllib.request as req

#html 가져오기
url='https://finance.naver.com/marketindex/'
res=req.urlopen(url)

#html 분석
soup=BeautifulSoup(res,'html.parser')

#원하는 데이터 추출
price =soup.select_one('#exchangeList > li.on > a.head.usd > div > span.value').string
print(price)
jp_price=soup.select_one('#worldExchangeList > li.on > a.head.jpy_usd > div > span.value').string
print(jp_price)