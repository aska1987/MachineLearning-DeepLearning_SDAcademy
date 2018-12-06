#html 페이지에는 상대주소로 되어있는 것을 절대 주소로 바꿔야 다운로드 가능
from urllib.parse import urljoin

base='http://example.com/html/a.html'

print(urljoin(base,'b.html'))
print(urljoin(base,'sub/c.html'))
print(urljoin(base,'../index.html'))
print(urljoin(base,'../img/hoge.png'))
print(urljoin(base,'../css/hoge.css'))


print('*'*20)

print(urljoin(base,'/hoge.html'))
print(urljoin(base,'http://otherExample.com/wiki'))
print(urljoin(base,'//anotherExample.org/test'))