# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:50:53 2018
선거 데이터
@author: SDEDU
"""

import pandas as pd
import numpy as np
import platform
import matplotlib.pyplot as plt

election_result=pd.read_csv('1130_data/election.csv',encoding='utf-8',index_col=0)

sido_candi=election_result['광역시도']
sido_candi=[name[:2] if name[:2]
in ['서울','부산','대구','광주','인천','대전','울산']
else '' for name in sido_candi]

def cut_char_sigu(name): #2글자 이름은 그대로, 3글자 이름은 줄이기위해서
    return name if len(name)==2 else name[:-1]

#광역시가 아닌데 행정구를 가지고 있는 수원, 성남,.. 에 대해 
import re
sigun_candi=['']*len(election_result)

for n in election_result.index:
    each=election_result['시군'][n]
    if each[:2] in ['수원','성남','안양','안산','고양','용인','청주','천안',
           '전주','포항','창원']:
        sigun_candi[n]=re.split('시',each)[0]+''+cut_char_sigu(re.split('시',each)[1])
    else:
        sigun_candi[n]=cut_char_sigu(each)

sigun_candi 

#sido_candi 와 위코드에서 정리한 시군구 이름이 저장된 변수 sigun_candi를 합침
#세종 예외처리
ID_candi=[sido_candi[n]+' ' +sigun_candi[n] for n in range(0,len(sigun_candi))]

ID_candi=[name[1:] if name[0]==' ' else name for name in ID_candi]
ID_candi=[name[:2] if name[:2]=='세종' else name for name in ID_candi]
ID_candi

election_result['ID']=ID_candi
election_result

election_result[['rate_moon','rate_hong','rate_ahn']]=election_result[['moon','hong','ahn']].div(election_result['pop'],axis=0)
election_result[['rate_moon','rate_hong','rate_ahn']]*=100

election_result.sort_values(['rate_moon'],ascending=[False]).head(10)
election_result.sort_values(['rate_hong'],ascending=[False]).head(10)
election_result.sort_values(['rate_ahn'],ascending=[False]).head(10)

draw_korea=pd.read_csv('1130_data/draw_korea.csv',encoding='utf-8',index_col=0)

#ID가 일치해야함
set(draw_korea['ID'].unique()) - set(election_result['ID'].unique())
set(election_result['ID'].unique()) - set(draw_korea['ID'].unique())

election_result[election_result['ID']=='고성']
election_result.loc[125,'ID']='고성(강원)'
election_result.loc[233,'ID']='고성(경남)'
election_result[election_result['시군']=='고성군']

election_result[election_result['광역시도']=='경상남도']

election_result.loc[228,'ID']='창원 합포'
election_result.loc[229,'ID']='창원 회원'

set(draw_korea['ID'].unique())-set(election_result['ID'].unique())
set(election_result['ID'].unique())-set(draw_korea['ID'].unique())

election_result[election_result['시군']=='부천시']

election_result.tail()


ahn_tmp=election_result.loc[85,'ahn']/3
hong_tmp=election_result.loc[85,'hong']/3
moon_tmp=election_result.loc[85,'moon']/3
pop_tmp=election_result.loc[85,'pop']/3

rate_moon_tmp=election_result.loc[85,'rate_moon']
rate_hong_tmp=election_result.loc[85,'rate_hong']
rate_ahn_tmp=election_result.loc[85,'rate_ahn']

election_result.loc[250]=['경기도','부천시',pop_tmp,
                   pop_tmp-(ahn_tmp+hong_tmp),ahn_tmp,hong_tmp,
                   '부천 소사',
                   rate_moon_tmp,rate_hong_tmp,rate_ahn_tmp]
election_result.loc[251]=[ '경기도','부천시',pop_tmp,
                    pop_tmp-(ahn_tmp+hong_tmp),ahn_tmp,hong_tmp,
                    '부천 오정',
                   rate_moon_tmp,rate_hong_tmp,rate_ahn_tmp]
election_result.loc[252]=['경기도','부천시',pop_tmp,
                    pop_tmp-(ahn_tmp+hong_tmp),ahn_tmp,hong_tmp,
                    '부천 원미',                   
                    rate_moon_tmp,rate_hong_tmp,rate_ahn_tmp]

election_result[election_result['시군']=='부천시']
election_result.drop([85],inplace=True)
election_result[election_result['시군']=='부천시']

set(draw_korea['ID'].unique())-set(election_result['ID'].unique())
set(election_result['ID'].unique())-set(draw_korea['ID'].unique())

final_elect_data=pd.merge(election_result, draw_korea,how='left',
                          on=['ID'])
final_elect_data.head()

final_elect_data['moon_vs_hong']=final_elect_data['rate_moon']-final_elect_data['rate_hong']
final_elect_data['moon_vs_anh']=final_elect_data['rate_moon']-final_elect_data['rate_ahn']
final_elect_data['ahn_vs_hong']=final_elect_data['rate_ahn']-final_elect_data['rate_hong']
final_elect_data

final_elect_data.sort_values(['moon_vs_hong'], ascending=True).head()





def drawKorea(targetData, blockedMap, d1, d2, cmapname):
    gamma = 0.75

    whitelabelmin = (max(blockedMap[targetData]) - min(blockedMap[targetData])) * 0.25 + min(blockedMap[targetData])

    datalabel = targetData

    vmin = min(blockedMap[targetData])
    vmax = max(blockedMap[targetData])

    BORDER_LINES = [
        [(3, 2), (5, 2), (5, 3), (9, 3), (9, 1)], # 인천
        [(2, 5), (3, 5), (3, 4), (8, 4), (8, 7), (7, 7), (7, 9), (4, 9), (4, 7), (1, 7)], # 서울
        [(1, 6), (1, 9), (3, 9), (3, 10), (8, 10), (8, 9),
         (9, 9), (9, 8), (10, 8), (10, 5), (9, 5), (9, 3)], # 경기도
        [(9, 12), (9, 10), (8, 10)], # 강원도
        [(10, 5), (11, 5), (11, 4), (12, 4), (12, 5), (13, 5),
         (13, 4), (14, 4), (14, 2)], # 충청남도
        [(11, 5), (12, 5), (12, 6), (15, 6), (15, 7), (13, 7),
         (13, 8), (11, 8), (11, 9), (10, 9), (10, 8)], # 충청북도
        [(14, 4), (15, 4), (15, 6)], # 대전시
        [(14, 7), (14, 9), (13, 9), (13, 11), (13, 13)], # 경상북도
        [(14, 8), (16, 8), (16, 10), (15, 10),
         (15, 11), (14, 11), (14, 12), (13, 12)], # 대구시
        [(15, 11), (16, 11), (16, 13)], # 울산시
        [(17, 1), (17, 3), (18, 3), (18, 6), (15, 6)], # 전라북도
        [(19, 2), (19, 4), (21, 4), (21, 3), (22, 3), (22, 2), (19, 2)], # 광주시
        [(18, 5), (20, 5), (20, 6)], # 전라남도
        [(16, 9), (18, 9), (18, 8), (19, 8), (19, 9), (20, 9), (20, 10)], # 부산시
    ]

    mapdata = blockedMap.pivot(index='y', columns='x', values=targetData)
    masked_mapdata = np.ma.masked_where(np.isnan(mapdata), mapdata)
    
    plt.figure(figsize=(8, 13))
    plt.pcolor(masked_mapdata, vmin=vmin, vmax=vmax, cmap=cmapname, edgecolor='#aaaaaa', linewidth=0.5)

    # 지역 이름 표시
    for idx, row in blockedMap.iterrows():
        annocolor = 'white' if row[targetData] > whitelabelmin else 'black'

        # 광역시는 구 이름이 겹치는 경우가 많아서 시단위 이름도 같이 표시한다. (중구, 서구)
        if row[d1].endswith('시') and not row[d1].startswith('세종'):
            dispname = '{}\n{}'.format(row[d1][:2], row[d2][:-1])
            if len(row[d2]) <= 2:
                dispname += row[d2][-1]
        else:
            dispname = row[d2][:-1]

        # 서대문구, 서귀포시 같이 이름이 3자 이상인 경우에 작은 글자로 표시한다.
        if len(dispname.splitlines()[-1]) >= 3:
            fontsize, linespacing = 9.5, 1.5
        else:
            fontsize, linespacing = 11, 1.2

        plt.annotate(dispname, (row['x']+0.5, row['y']+0.5), weight='bold',
                     fontsize=fontsize, ha='center', va='center', color=annocolor,
                     linespacing=linespacing)
        
    # 시도 경계 그린다.
    for path in BORDER_LINES:
        ys, xs = zip(*path)
        plt.plot(xs, ys, c='black', lw=4)

    plt.gca().invert_yaxis()
    #plt.gca().set_aspect(1)

    plt.axis('off')

    cb = plt.colorbar(shrink=.1, aspect=10)
    cb.set_label(datalabel)

    plt.tight_layout()
    plt.show() 

    
    
import folium
import json

pop_folium=final_elect_data.set_index('ID')
del pop_folium['광역시도']
del pop_folium['시군']    

geo_path='1130_data/skorea_geo_simple.json'
geo_str=json.load(open(geo_path,encoding='utf-8'))

map=folium.Map(location=[36.2002,127.054], zoom_start=6)
map.choropleth(geo_data=geo_str,
               data=pop_folium['moon_vs_hong'],
               columns=[pop_folium.index,pop_folium['moon_vs_hong']],
               fill_color='YlGnBu',
               key_on='feature.id')
map.save('map.html')

