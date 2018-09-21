# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:38:11 2018

@author: SDEDU
"""
#나이 계산
sampleData=[1990,1991,1994,1990,1992,1996]
age=[]
for i in sampleData:
    age.append(2018-i)
print(age)

#list
listData=[1,2,-8]
sum(listData)
listData=[1,2,-8,0]
max(listData)
listSum=0
for i in listData:
    listSum+=i
listSum
maxValue=0
for i in listData:
    if i>maxValue:
        maxValue=i
maxValue
#Investment Program
Capital=float(input('Enter the investment amount: '))
years=int(input('Enter the number of years: '))
rate=float(input('Enter the rate as a %: '))

if Capital>0 and years>0 and rate>0:
    initialC=Capital
    print('Year /Starting balance /Interest /Ending balance')
    for i in range(1,years+1):
        print('%d     %.2f \t \t %.2f   %.2f' %(i,Capital,Capital*(rate/100),Capital+Capital*(rate/100)))
        Capital+=Capital*(rate/100)
        if i==(years):
            print('Ending balance: $%.2f' %Capital)
            print('Total interest earned : $%.2f' %(Capital-initialC))
else:
    print('enter is invalid')

a = 33
b = 200
if b > a:
  print("b is greater than a")
  
a = 33
b = 33
if b > a:
  print("b is greater than a")
elif a == b:
  print("a and b are equal")
  
a = 200
b = 33
if b > a:
  print("b is greater than a")
elif a == b:
  print("a and b are equal")
else:
  print("a is greater than b")
  
i = 1
while i < 6:
  print(i)
  if i == 3:
    break
  i += 1

i = 0
while i < 6:
  i += 1 
  if i == 3:
    continue
  print(i)
  
print("Sammy has {} balloons.".format(5))

open_string = "Sammy loves {}."
print(open_string.format("open source"))

print("Sammy is a {3}, {2}, and {1} {0}!".format("happy", "smiling", "blue", "shark"))

print("Sammy the {pr} {1} a {0}.".format("shark", "made", pr = "pull request"))
print("Sammy has {0:<4} red {1:^16}!".format(5, "balloons"))
print("{:*^20s}".format("Sammy"))

print('Hello {0} {1}. Your current balance is ${2}'.format('John','Doe',53.44))
