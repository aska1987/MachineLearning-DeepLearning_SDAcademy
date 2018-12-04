import urllib.request
import urllib.parse

'''
웹서비스하는 서버에 인자를 전달하는 방법

영문으로 되어있을 때는 아래 코드와 같이 전달해도 된다. 
API='http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=109'

한글일 경우는 웹에서 데이터가 변환되므로(encoding) 이렇게 전달하면 안된다.
API='http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?'
values={'stdId':'109'}
params=urllib.parse.urlencode(values)
url=API+'?'+params
print('url=',url)

'''
API='http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=109'
data=urllib.request.urlopen(API).read()
text=data.decode('utf-8')
print(text)

