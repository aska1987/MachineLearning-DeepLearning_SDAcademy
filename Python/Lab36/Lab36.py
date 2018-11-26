# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 09:43:37 2018
자연어 처리
유사도
@author: SDEDU
"""

#문장의 유사도 측정하기
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(min_df=1)
contents=['메리랑 놀러가고 싶지만 바쁜데 어떻하죠?',
          '메리는 공원에서 산책하고 노는 것을 싫어해요',
          '메리는 공원에서 노는 것도 싫어해요. 이상해요.',
          '먼 곳으로 여행을 떠나고 싶은데 너무 바빠서 그러질 못하고 있어요']

x=vectorizer.fit_transform(contents)
vectorizer.get_feature_names() # 첫번째 요소: '것도'
x.toarray().transpose() # 첫번째 요소 '것도'는 contents의 3번째 요소에만 있음

## 형태소 분석 ############################3
from konlpy.tag import Okt
t=Okt()
contents_tokens=[t.morphs(row) for row in contents]
contents_tokens
#--> 형태소 분석 결과 '메리랑','메리는' 을 '메리' 로 분리해서 같은 단어로 보고있다. 

contents_for_vectorize=[]
for content in contents_tokens:
    sentence=''
    for word in content:
        sentence = sentence + ' ' + word
        
    contents_for_vectorize.append(sentence)
contents_for_vectorize

x=vectorizer.fit_transform(contents_for_vectorize)
num_samples,num_features=x.shape
num_samples,num_features

new_post=['메리랑 공원에서 산책하고 놀고 싶어요']
new_post_tokens=[t.morphs(row) for row in new_post]
new_post_for_vectorize=[]
for content in new_post_tokens:
    sentence =''
    for word in content:
        sentence=sentence+ ' ' +word
    new_post_for_vectorize.append(sentence)
new_post_for_vectorize

new_post_vec=vectorizer.transform(new_post_for_vectorize)
new_post_vec.toarray()


## 새로운 문장(new_post_vec)과 비교해야 할 문장(contents)들에 각각에 대해 거리를 구함
import scipy as sp
def dist_raw(v1,v2):
    delta=v1-v2
    return sp.linalg.norm(delta.toarray())
best_doc=None
best_dist=65535
best_i=None

for i in range(0,num_samples):
    post_vec=x.getrow(i)
    d=dist_raw(post_vec,new_post_vec)
    
    print('== Post %i with dist=%.2f   : %s' %(i,d,contents[i]))
    if d<best_dist:
        best_dist=d
        best_i=i

print('Best post is %i, dist=%.2f' %(best_i,best_dist))
print('-->',new_post)
print('---->',contents[best_i])
for i in range(0,len(contents)):
    print(x.getrow(i).toarray())
print('-------------')
print(new_post_vec.toarray())    
'''
텍스트마이닝에서 사용하는 단어별로 부과하는 가중치
tf(term frequency): 단어가 문서내에서 자주 등장할수록 중요도가 높을 것으로 봄
idf(inverse document frquency): 핵심어휘일지는 모르지만 문서 간의 비교에서는 중요한 단어가 아니라는 뜻)
'''
    
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(min_df=1,decode_error='ignore')
contents_tokens=[t.morphs(row) for row in contents]
contents_for_vectorize=[]
for content in contents_tokens:
    sentence=''
    for word in content:
        sentence =sentence+ ' '+word
    contents_for_vectorize.append(sentence)
    
x=vectorizer.fit_transform(contents_for_vectorize)
num_samples,num_features=x.shape
num_samples,num_features

new_post=['근처 공원에 메리랑 놀러가고 싶네요.']
new_post_tokens=[t.morphs(row) for row in new_post]
new_post_for_vectorize=[]
for content in new_post_tokens:
    sentence=''
    for word in content:
        sentence= sentence+' ' +word
    new_post_for_vectorize.append(sentence)
new_post_for_vectorize

new_post_vec=vectorizer.transform(new_post_for_vectorize)
best_doc=None
best_dist=65535
best_i=None

for i in range(0,num_samples):
    post_vec=x.getrow(i)
    d=dist_raw(post_vec,new_post_vec)
    
    print('== Post %i with dist=%.2f   : %s' %(i,d,contents[i]))
    if d<best_dist:
        best_dist=d
        best_i=i

print('Best post is %i, dist=%.2f' %(best_i,best_dist))
print('-->',new_post)
print('---->',contents[best_i])
