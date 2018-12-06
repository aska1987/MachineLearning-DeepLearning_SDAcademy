from bs4 import BeautifulSoup

html='''
<html>
    <body>
        <div id='meigen'>
            <h1>의적</h1>
            <ul class='items'>
                <li>홍길동</li>
                <li>꺽정</li>
                <li>길산</li>
            </ul>
        </div>
    </body>
</html>
'''

soup=BeautifulSoup(html,'html.parser')
h1=soup.select_one('div#meigen > h1').string
print(h1)

#목록 추출
li_list=soup.select('div#meigen > ul.items > li')
for li in li_list:
    print(li)