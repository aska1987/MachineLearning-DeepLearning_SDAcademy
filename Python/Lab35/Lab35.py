# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 09:41:44 2018
교차 검증
자연어 처리
@author: SDEDU
"""
'''
교차 검증
훈련세트와 테스트세트로 한번나누는것보다 더 안정적이고 뛰어난 통계적 평가 방법
단점: 모델 k개를 만들어야 하므로 k배 느려짐

'''
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

x,y=make_blobs(random_state=0)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)

#모델 객체를 만들고 훈련세트로 학습
logreg=LogisticRegression().fit(x_train,y_train)
logreg.score(x_test,y_test)

## scikit-learn의 교차검증 ################################
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris=load_iris()
logreg=LogisticRegression()

scores=cross_val_score(logreg,iris.data,iris.target)
scores #교차검증 점수
#cross_val_score의 기본값은 3겹 교차검증으므로 정확도 값이 3개가 반환 
#cv매개변수로 기본값 수정가능
cross_val_score(logreg,iris.data,iris.target,cv=5)
cross_val_score(logreg,iris.data,iris.target,cv=5).mean() #교차검증 평균점수


'''
KoNLPy : 한국어 자연어처리 위한 대표적인 라이브러리
NLTK : 영어로된 텍스트의 자연어처리를 위한 대표적인 라이브러리
'''

## Naive Bayes Classifier  #######################
#영어
from nltk.tokenize import word_tokenize
import nltk
train=[('i like you','pos'),
       ('i hate you','neg'),
       ('you like me','neg'),
       ('i like her','pos')]
all_words=set(word.lower() for sentence in train
              for word in word_tokenize(sentence[0]))
all_words #말뭉치

t=[({word:(word in word_tokenize(x[0])) for word in all_words},x[1])
for x in train]
t

classifier=nltk.NaiveBayesClassifier.train(t)
classifier.show_most_informative_features() 
# -->ex) me 가 없을 때 'pos'가 나타날 확률은 'neg' 보다 1.7 배 더 높다

test_sentence='i like MeRui'
test_sent_features={word.lower():
    (word in word_tokenize(test_sentence.lower()))
    for word in all_words}
test_sent_features
classifier.classify(test_sent_features)

#한글
from konlpy.tag import Twitter
pos_tagger=Twitter()

train=[('메리가 좋아','pos'),
       ('고양이도 좋아','pos'),
       ('난 수업이 지루해','neg'),
       ('메리는 이쁜 고양이야','pos'),
       ('난 마치고 메리랑 놀거야','pos')]
all_words=set(word.lower() for sentence in train
              for word in word_tokenize(sentence[0]))
all_words 

t=[({word: (word in word_tokenize(x[0])) for word in all_words}, x[1])
for x in train]
t

test_sentence='난 수업이 마치면 메리랑 놀거야'
test_sent_features={word.lower():
    (word in word_tokenize(test_sentence.lower()))
    for word in all_words}
test_sent_features
    
classifier.classify(test_sent_features)

def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True,stem=True)]

train_docs=[(tokenize(row[0]),row[1]) for row in train]
train_docs    

tokens=[t for d in train_docs for t in d[0]]
tokens

def term_exists(doc):
    return {word:(word in set(doc)) for word in tokens}
train_xy=[(term_exists(d),c) for d,c in train_docs]
train_xy

classifier=nltk.NaiveBayesClassifier.train(train_xy)
test_sentence=[('난 수업이 마치면 메리랑 놀거야')]
test_docs=pos_tagger.pos(test_sentence[0])
test_docs
classifier.show_most_informative_features()

test_sent_features={word:(word in tokens) for word in test_docs}
test_sent_features

classifier.classify(test_sent_features)

import pandas as pd
import wordcloud
#영문
f=open('C:\\Users\\SDEDU\\Documents\\GitHub\\MachineLearning-DeepLearning_SDAcademy\\Python\\Lab35\\test.txt')
data=f.read()
data
token=nltk.word_tokenize(data)
token
test=nltk.Text(token,name='test')
test.plot()


voca_most=test.vocab()
wc=wordcloud.WordCloud(relative_scaling=0.2,
                       ).generate_from_frequencies(dict(voca_most))
plt.imshow(wc)



wc=wordcloud.WordCloud().generate(data)
wc.words_
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')


#한글
f=open('20대창업gsub.txt')
data=f.read()
data=data.replace('\n',' ')
token=nltk.word_tokenize(data)
test=nltk.Text(token,name='창업')

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
test.plot()

voca_most=test.vocab()
wc=wordcloud.WordCloud(font_path='c:\\Windows\\Fonts\\malgun.ttf',
                       relative_scaling=0.2).generate_from_frequencies(dict(voca_most))
plt.imshow(wc)

wc=wordcloud.WordCloud(font_path='c:\\Windows\\Fonts\\malgun.ttf').generate(data)
wc.words_
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
