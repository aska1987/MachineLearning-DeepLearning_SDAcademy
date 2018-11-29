# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 14:39:01 2018

@author: SDEDU
"""

from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd

df=pd.read_csv('best_sandwiches_chicago.csv',index_col=0)


html=urlopen(df['URL'][0])
soup_tmp=BeautifulSoup(html,'html.parser')
soup_tmp

print(soup_tmp.find('p','addy'))

price_tmp=soup_tmp.find('p','addy').get_text()
price_tmp

price_tmp.split()

price_tmp.split()[0] # 가격
price_tmp.split()[0][:-1] # . 제거
' '.join(price_tmp.split()[1:-2])

price=[]
address=[]

for n in df.index:
    html=urlopen(df['URL'][n])
    soup_tmp=BeautifulSoup(html,'lxml')
    
    gettings=soup_tmp.find('p','addy').get_text()
    price.append(gettings.split()[0][:-1])
    address.append(' '.join(gettings.split()[1:-2]))
    
len(price)
len(address)
len(df)
df['Price']=price
df['Address']=address
df=df.loc[:,['Rank','Cafe','Menu','Price','Address']]
df.set_index('Rank',inplace=True)

import folium
import pandas as pd
import googlemaps
import numpy as np


lat=[]
lng=[]


for n in df.index:
    if df['Address'][n] !='Multiple':
        target_name=df['Address'][n]+'. '+'Cicago'
        gmaps_output=googlemaps.geocoding(target_name)
        location_output=gmaps_output[0].get('geometry')
        lat.append(location_output['location']['lat'])
        lng.append(location_output[''])
        

