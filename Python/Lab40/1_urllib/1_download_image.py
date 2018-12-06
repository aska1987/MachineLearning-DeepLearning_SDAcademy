#urlib 는 웹에서 데이터를 다운로드할때 사용
#BeautifulSoup 은 웹에 html이나 xml를 구조화 후 추출할때 사용

import urllib.request

url='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSzWs2ON620KNi2e0rJsBX_ry90y3_ValLBf8DRBw-L4-Ezjr_N'
savename='아이린.jpg'

# url='data:image/jpeg;base64,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'
# savename='트와이스.jpg'

#다운로드
urllib.request.urlretrieve(url,savename)
print('저장완료')
