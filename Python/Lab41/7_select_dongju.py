from bs4 import BeautifulSoup
import urllib.request as req

url='https://ko.wikisource.org/wiki/%EC%A0%80%EC%9E%90:%EC%9C%A4%EB%8F%99%EC%A3%BC'
res=req.urlopen(url)

def selFunc(css_sel):
    a_list=soup.select(sel)
    for a in a_list:
        name=a.string
        print('-',name)
soup=BeautifulSoup(res,'html.parser')


#bs4 에서 nth-child는 지원하지 않는다
sel='#mw-content-text > div > ul:nth-child(6) > li > ul > li:nth-child(1) > a'
sel='#mw-content-text > div > ul > li > ul > li > a'
selFunc(sel)
print('-'*20)
sel='#mw-content-text > div > ul > li > a'
selFunc(sel)

