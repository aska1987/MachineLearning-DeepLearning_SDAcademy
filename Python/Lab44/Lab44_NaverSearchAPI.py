# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 10:34:16 2018

@author: SDEDU
"""
'''
서비스 웹사이트에서 제공해주는 기능
OpenAPI
send : URL + 약속된 인잔
receive : 약속된 데이터
해당 서비스 사이트에 문서를 읽어서 약속대로 보내야 응답 받음
'''

# 네이버 검색 API예제는 블로그를 비롯 전문자료까지 호출방법이 동일하므로 blog검색만 대표로 예제를 올렸습니다.
# 네이버 검색 Open API 예제 - 블로그 검색
import os
import sys
import urllib.request
import datetime
import time
import json
from config import *
client_id = "_M6QMrUwSzIQrzLnWOMj"
client_secret = "MdCHqJzc1D"

    
def getPostData(post,jsonResult):
    title=post['title']
    description=post['description']
    org_link=post['originallink']
    link=post['link']
    pDate=datetime.datetime.strptime(post['pubDate'],
                                     '%a, %d %v %Y %H:%M:%S +0900')
    pDate=pDate.strftime('%Y-%m-%d %H:%M:%S')
    jsonResult.append({'title':title,'description':description,
                       'org_link':org_link,'link':link,
                       'pDate':pDate})
    

def get_request_url(url):
    req=urllib.request.Request(url)
    req.add_header("X-Naver-Client-Id",client_id)
    req.add_header("X-Naver-Client-Secret",client_secret)
    #print('get_requset_url 진입!')
    try:
        response=urllib.request.urlopen(req)
        print(response.getcode())
        if response.getcode() == 200:
            print('getcode = 200')
            print('[%s] Url Response Success' % datetime.datetime.now())
            return response.read().decode('utf-8')
    except Exception as e:
        print(e)
        print
        print('[%s] Error for URL: %s' % (datetime.datetime.now(),url))
        return None
    
def getNaverSearchResult(sNode,search_text,page_start,display):
    #print('getNaverSearchResult 진입!')
    base='https://openapi.naver.com/v1/search'
    node='/%s.json' % sNode 
    parameters= '?query=%s&start=%s&display=%s' %(
            urllib.parse.quote(search_text),page_start,display)
    url=base+node+parameters
    retData=get_request_url(url)
    if(retData==None):
        return None
    else:
        return json.loads(retData)
    

def main():
    while True:
        
        jsonResult=[] #검색을 저장
        sNode='news'
        search_text=input('검색할 단어를 입력하세요: ')
        if search_text=='quit':
            break
        print('검색 단어:',search_text)
        display_count=100
        jsonSearch=getNaverSearchResult(sNode,search_text,1,display_count)
        
        while((jsonSearch != None) and (jsonSearch['display'] !=0)):
            for post in jsonSearch['items']:
                getPostData(post,jsonResult)
                
            nStart=jsonSearch['start'] + jsonSearch['display']
            jsonSearch=getNaverSearchResult(
                    sNode,search_text,nStart,display_count)
        with open('%s_naver_%s.json' %(search_text,sNode),'w',
                  encoding='utf-8') as outfile:
            retJson=json.dumps(jsonResult,indent=4,
                               sort_keys=True,
                               ensure_ascii=False)
            outfile.write(retJson)
        print('%s_naver_%s.json SAVED'%(search_text,sNode))
        
    print('검색종료')
if __name__=='__main__':
    main()
