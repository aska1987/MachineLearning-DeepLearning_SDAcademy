# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 15:02:50 2018

@author: SDEDU
"""
import sys
sys.argv

import os
filename = os.path.splitext("points.txt")
filename[1]

import random
random.randint(2,6)

from math import pi,sqrt
print(pi,sqrt(2))
#정규식
import re
m=re.search('(?<=abc)def', 'abcdef')
m.group(0)

m=re.search(r'(?<=-)\w+','spam-egg')
m.group(0)


pattern = re.compile("d")
pattern.search("dog")
pattern.search("dog", 1)

 pattern = re.compile("o")
pattern.match("dog") 
pattern.match("dog", 1)

re.match("c", "abcdef")
re.search("c", "abcdef")  

re.match("c", "abcdef")
re.search("^c", "abcdef")
re.search("^a", "abcdef")

#모듈 
import mymodule
mymodule.greeting('Lee')
mymodule.person1['age']

import datetime
x = datetime.datetime.now()
print(x)

#file I/O
f=open('mytext.txt','r')
f.readline()
#Create any text file and write a program to read an entire text file.
open('mytext.txt','r').read()
#Write a program to read a file line by line and store it into a list.
f=open('mytext.txt','r')
storeLine=[]
for i in f:
    storeLine.append(f.readline())
storeLine

with open('mytext.txt','r') as file:
    storeLine=file.readlines()
storeLine
#Write a program to read a file line by line store it into a variable.
f=open('mytext.txt','r')
storeVariable=''
for i in f:
    storeVariable+=f.readline()
storeVariable
#Extra: Count the number of words in each line
for i in storeLine:
    print(len(i.split(' ')))
#Open any text file and count the number of sentence, words, and charactors.
print(len(storeVariable.split('. '))) #sentence
print(len(storeVariable.split(' '))) #words
print(len(storeVariable)) #charactors

