# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 09:51:25 2018

@author: SDEDU
"""


import pandas as pd

code_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13', header=0)[0] 
# 종목코드가 6자리이기 때문에 6자리를 맞춰주기 위해 설정해줌 
code_df.종목코드 = code_df.종목코드.map('{:06d}'.format) 
# 우리가 필요한 것은 회사명과 종목코드이기 때문에 필요없는 column들은 제외해준다. 
code_df = code_df[['회사명', '종목코드']] 
# 한글로된 컬럼명을 영어로 바꿔준다. 
code_df = code_df.rename(columns={'회사명': 'name', '종목코드': 'code'}) 
code_df.head()


def get_url(item_name,code_df):
    code=code_df.query("name=='{}'".format(item_name))['code'].to_string(index=False)
    url='http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code)
    print('요청 URL={}'.format(url))
    return url

item_name='에프앤리퍼블릭'
url=get_url(item_name,code_df)

df=pd.DataFrame()

for page in range(1, 100): 
    pg_url = '{url}&page={page}'.format(url=url, page=page)
    df = df.append(pd.read_html(pg_url, header=0)[0], ignore_index=True) 

df=df.dropna()

from yahoo_finance import Share
