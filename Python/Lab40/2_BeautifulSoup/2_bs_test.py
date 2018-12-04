from bs4 import BeautifulSoup
html="""
<html>
    <body>
        <h1 id='title'>스크래핑</h1>
        <p>웹분석</p>
        <p id='body'>원하는 정보 추출</p>
    </body>
</html>
"""
#구조적 분석
soup=BeautifulSoup(html,'html.parser')

#원하는 정보 추출

title=soup.find(id='title')
body=soup.find(id='body')

#텍스트만 출력
print("title=", title.string)
print("body=", body.string)