import urllib.request

#웹에 접속해서 정보 다운로드 하기
url='http://www.naver.com'
res=urllib.request.urlopen(url)
data=res.read()

#웹의 데이터를 우리가 알아볼 수 있게 변환하기
text=data.decode('utf-8')
print(text)

savename='naver.txt'
with open(savename,mode='wt') as f:
    f.write(text)
    print('저장완료')