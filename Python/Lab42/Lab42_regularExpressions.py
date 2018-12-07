# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 12:50:20 2018

regularExpressions

https://wikidocs.net/4308 참조

@author: SDEDU
"""

data='''
park 800905-1049118
kim 700905-1059119
'''
print(data)

#주민번호 앞자리만 추출
import re
pat=re.compile('(\d{6})[-]\d{7}')
print(pat.sub('\g<1>-*******',data))
