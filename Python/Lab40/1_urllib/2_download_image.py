import urllib.request

url='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRfcqTBBa6h-gL4gPG7IkC7V9jHCt7lWJ8SvGdNNlRKOdbfG1SF'
savename='자연.jpg'
#다운로드
mem=urllib.request.urlopen(url).read()
#파일로 저장
with open(savename, mode='wb') as f:
    f.write(mem)
    print('저장완료')
