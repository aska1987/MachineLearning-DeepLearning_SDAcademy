from bs4 import BeautifulSoup

#분석하고 싶은 html

html="""
<html>
    <body>
        <h1>Scraping</h1>
        <p>Analysis Web</p>
        <p>Crop Data</p>
    </body>
</html>
"""

#구조적으로 분석하기
soup=BeautifulSoup(html,'html.parser')

# 원하는 부분 추출하기
h1=soup.html.body.h1
p1=soup.html.body.p
p2=p1.next_sibling.next_sibling

print('h1=',h1.string)
print('p1=',p1.string)
print('p2=',p2.string)

