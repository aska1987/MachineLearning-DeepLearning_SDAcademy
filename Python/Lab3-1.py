# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 09:35:48 2018

@author: SDEDU
"""
#함수 만들기
def my_function():
    print('Hello from a function')
def my_function(fname):
    print(fname+' Lee')
my_function('jae')

def changeme(mylist):
    mylist.append([1,2,3,4])
    print('Values inside the function: ',mylist)
    return
Mylist=[10,20,30]
changeme(mylist)

def changeme(mylist):
    mylist=[1,2,3,4]
    print('Values inside the function:', mylist)
    return
mylist=[10,20,30]
changeme(mylist);

def gcd(a,b):
    while a !=0:
        a,b =b%a,a
    return b
gcd(12,20)
#화씨에서 썹씨로 
def calculatesDegrees(Fahrenheit):
    return ((Fahrenheit-32)/1.8)
calculatesDegrees(5)
#list에 모든 값을 곱한 값
def multiplyList(mList):
    multiplyValue=1
    for i in mList:
        multiplyValue*=i;
    return multiplyValue
mList=[8,2,3,-1,7]
multiplyList(mList)
#문장 단어 거꾸로 출력
def reverseWords(words):
    temp=[]
    temp=words.split(' ')
    temp.reverse()
    return print(" ".join(temp))
words=input('문장을 입력하세요: ')
reverseWords(words)

#join
listofstrings=['a','b','c']
result='**'.join(listofstrings)
result
#lambda
x = lambda a : a + 10
print(x(5))

x = lambda a, b : a * b
print(x(5, 6))

x = lambda a, b, c : a + b + c
print(x(5, 6, 2))

def myfunc(n):
  return lambda a : a * n
mydoubler = myfunc(2)
print(mydoubler(11))

def myfunc(n):
  return lambda a : a * n
mydoubler = myfunc(2)
mytripler = myfunc(3)
print(mydoubler(11)) 
print(mytripler(11))

def gameBoard(width,column):
    num=1;
    for i in range(1,width+1):
        if num%2==1:
            for j in range(1,column+1):
                print ('---',end=" ")
        else:
            for j in range(0,column+1):
                print ('|  ',end=" ")            
        num+=1
        print()
gameBoard(7,3)

