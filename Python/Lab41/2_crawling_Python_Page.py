from bs4 import BeautifulSoup
from urllib.request import *
from urllib.parse import *
from os import makedirs
import os.path, time, re

# 중복 방지 파일
proc_files={}

#HTML 내부에 있는 링크 추출
def enum_links(html,base):
    soup=BeautifulSoup(html,'html.parser')
    links=soup.select('link[rel="stylesheet"]') #CSS
    links +=soup.select('a[href]') #링크
    result=[]
    #href 속성을 추출하고, 링크를 절대 경로로 변환
    for a in links:
        href=a.attrs['href']
        url=urljoin(base,href)
        result.append(url)
    return result
    pass
#파일을 다운받고 저장하는 함수
def download_file(url):
    o=urlparse(url)
    savepath='./' +o.netloc+o.path
    if re.search(r'/$',savepath): #폴더라면
        savepath +='index.html'
    savedir=os.path.dirname(savepath)
    if os.path.exists(savepath):
        return savepath
    if not os.path.exists(savedir):
        print('mkdir = ',savedir)
        makedirs(savedir)
    #파일 다운로드 하기
    try:
        print('download=',url)
        urlretrieve(url,savepath)
        time.sleep(1)
        return savepath
    except:
        print('다운 실패:',url)
        return None

    pass
#HTML 분석하고 다운받는 함수
def analyze_html(url,root_url):
    save_path=download_file(url)
    if save_path is None:
        return
    if save_path in proc_files:
        return
    proc_files[save_path]=True
    print('analyze_html =',url)
    #링크 추출
    html=open(save_path, 'r', encoding='utf-8').read()
    links=enum_links(html,url)

    for link_url in links:
        #다른 사이트에 있는 파일이면서 css가 아닐떄
        if link_url.find(root_url) !=0:
            if not re.search(r'.css$',link_url):
                continue
        #base url 주소를 가진 서버의 html파일이라면
        if re.search(r'.(html|htm)$', link_url):
            #재귀적으로 분석을 한다
            analyze_html(link_url,root_url)
            continue
        #기타 파일
        download_file(link_url)
    pass

'''
일반적으로 파이선 프로그램의 시작부분이다.
python 프로그램
1) 실행파일 : 직접실행
   __name__=='__main__'

2) 모듈(라이브러리) : 간접 실행(import)
   __name__=='모듈 이름'   
'''

if __name__ =='__main__':
    url = 'https://docs.python.org/3.5/library/'
    analyze_html(url,url)
