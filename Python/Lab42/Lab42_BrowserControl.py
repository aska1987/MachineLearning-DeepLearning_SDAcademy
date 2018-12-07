# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 09:46:46 2018

Browser Control
Selenium, chromeDriver, phantomjs
@author: SDEDU
"""

'''
Selenium
- 1) pip install Selenium
- 2) Web에서 chromeDriver 설치
- 3) 제어한다

urllib으로는 정해진 html에서만 가져올 수 밖에 없다
but 브라우저를 제어하면 javascript에 의해 동적인 html도 가져올 수 있음

항목(태그) 1번째 요소 찾기
find_element_by_id(id)
find_element_by_name(name)
find_element_by_css_selector(선택자)
find_element_by_xpath(xpath) : 많이쓰는편
find_element_by_tag_name(tag명)
find_element_by_link_text(link텍스트)
find_element_by_partial_link_text(text) : 링크의 자식 요소에 포함된 테스트로 
find_element_by_class_name(클래스명)

항목 모두 찾기
find_elements_by_css_selector(선택자)
find_elements_by_xpath(xpath) : 많이쓰는편
find_elements_by_tag_name(tag명)
find_elements_by_partial_link_text(text) : 링크의 자식 요소에 포함된 테스트로 
find_elements_by_class_name(클래스명)

항목을 찾은 후 명령하기
- clear() : 글자 지우기
- click() : 누르기
- get_attribute(name) : name에 해당하는 속성 추출
- is_displayer() : 요소가 화면에 출력되는지 확인
- is_enable() : 요소가 활성화 되어 있는지 확인
- is_selected() : 체크박스등이 선택된 상태인지 확인
- screenshot(filename) : 스크린샷 찍기
- send_keys(value) : 키를 입력하기
- submit() : 입력양식 전송
- value_of_css_property(name) : name에 해당하는 CSS속성의 값 추출
- id : 항목의 id 값
- location : 항목의 위치
- parent : 항목의 부모항목
- rect : 크기와 위치 정보를 가진 딕셔너리형을 리턴
- screenshot_as_base64 : 스크린샷을 Base64로 추출
- screenshot_as_png : 스크린샷을png로 추출
- size : 항목의 크기
- tag_name : 태그 이름
- text : 항목 내부의 글자

PhantomJS 브라우저 기준
- back() / forward() : 이전페이지/ 다음페이지
- close() : 브라우저 닫기
- current_url : 현재 url 추출
- get(url) : url을 읽어서 html을 리턴
- implicity_wait(sec) : 최대 대기시간(초) == time.sleep(sec)
- quit() : 드라이버 종료 -> 브라우저 닫힘
- save_screenshot(filename) : 스크린샷 저장
- set_window_size(width, height, windowHandle='current') : 브라우저 크기
- title : 현재 페이지의 타이틀 추출
- set_page_load_timeout(time_to_wait) : 페이지를 읽는 타임아웃시간 지정
  등등
'''
## naver 를 이용한 html 제어 #######################################
from selenium import webdriver
import time
ch=webdriver.Chrome('chromedriver.exe')
ch.get('http://www.naver.com') #네이버 출력
#ch.save_screenshot('images/001.jpg') #스샷
time.sleep(5)

num=0
def captureImage():
    global num
    imageName='images/' + '%03d' %(num) + '.png'
    ch.save_screenshot(imageName)
    num +=1
    
captureImage()
captureImage()
captureImage()
captureImage()
captureImage()

#네이버 홈페이지에서 인공지능 검색
edit_search=ch.find_element_by_xpath("""//*[@id="query"]""")
edit_search.clear()
edit_search.send_keys('인공지능')

btn_search=ch.find_element_by_xpath("""//*[@id="search_btn"]""")
btn_search.click()
ch.save_screenshot('images/인공지능.jpg') #스샷
time.sleep(1)


#인공지능 검색한 페이지에서 딥러닝 검색
edit_search=ch.find_element_by_xpath("""//*[@id="nx_query"]""") 
edit_search.clear()
edit_search.send_keys('딥러닝')

btn_search=ch.find_element_by_xpath("""//*[@id="nx_search_form"]/fieldset/button""")
btn_search.click()
ch.save_screenshot('images/딥러닝.jpg') #스샷


## opinet 에서 데이터 자동으로 받기 ####################################
num=5
def captureImage(browser):
    global num
    imageName='images/'+'%03d' %(num) + '.png'
    browser.save_screenshot(imageName)
    num+=1
def save_html(fileName,html):
    with open(fileName,'wt',encoding='utf-8') as f:
        f.write(html)

ch=webdriver.Chrome('chromedriver.exe')
ch.get('http://www.opinet.co.kr/searRgSelect.do')
time.sleep(1)
captureImage(ch)
        
##시도 지역 받아오기 ###################
sido_list_raw=ch.find_element_by_xpath("""//*[@id="SIDO_NM0"]""")
sido_list=sido_list_raw.find_elements_by_tag_name('option')
print(sido_list)
#1) value 값으로 얻기
sido_name=[option.get_attribute("value") for option in sido_list]
print(sido_name)
#2) text 값으로 얻기
sido_name=[option.text for option in sido_list]
print(sido_name)
sido_name.remove('시/도')
print(sido_name)

element=ch.find_element_by_xpath('''//*[@id="SIDO_NM0"]''')
element.send_keys(sido_name[0])

#시도 지역 자동으로 보내기
for i in range(0,len(sido_name)):
    element=ch.find_element_by_xpath('''//*[@id="SIDO_NM0"]''')
    element.send_keys(sido_name[i])
    time.sleep(1)

## 구 지역 받아오기#################
gu_list_raw=ch.find_element_by_xpath('''//*[@id="SIGUNGU_NM0"]''')
gu_list=gu_list_raw.find_elements_by_tag_name('option')
#1) value 값 얻기
gu_name=[option.get_attribute('value') for option in gu_list]
print(gu_name)
#2) text 값 얻기
gu_name=[option.text for option in gu_list]
gu_name

gu_name.remove('시/군/구')
print(gu_name)

## seoul 시군구 데이터 text , html 파일 받기  ###
for name in gu_name:
    gu_list_raw=ch.find_element_by_xpath('''//*[@id="SIGUNGU_NM0"]''')
    gu_list_raw.send_keys(name) #시군구 선택에서 name에 해당하는 곳으로 send key
    time.sleep(1)
    get_excel_btn=ch.find_element_by_xpath('''//*[@id="glopopd_excel"]''')
    get_excel_btn.click() #엑셀 저장 버튼 클릭
    htmlFile=ch.page_source
    #save_html('SeoulTextData/'+name+'.txt',htmlFile) #txt 파일로 저장
    save_html('SeoulHtmlData/'+name+'.html',htmlFile) #txt 파일로 저장
    captureImage(ch) #해당 웹 캡처
    print('save:',name)
    time.sleep(1)


'''
PhantomJS
- 눈에 안보이는 브라우저를 명령으로 제어할 수 있다.
- JavaScript를 지원함
'''
phtm=webdriver.PhantomJS('phantomjs/bin/phantomjs')
phtm.get('http://www.naver.com')
phtm.save_screenshot('images/naver.png')
time.sleep(1)
phtm.get('http://www.daum.net')
phtm.save_screenshot('images/daum.png')
time.sleep(1)
phtm.get('http://www.google.co.kr')
phtm.save_screenshot('images/google.png')
time.sleep(1)
phtm.get('http://www.nate.com')
phtm.save_screenshot('images/nate.png')

html=phtm.page_source
print(html)

