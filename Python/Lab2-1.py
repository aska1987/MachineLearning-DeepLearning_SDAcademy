# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
name='john'
print("hello, %s!" % name)

age=23
print('%s is %d years old' %(name,age))

import math
pi=math.pi
print('%f' %pi)
print('%.3f' %pi)
print('%1.0f' %pi)
print('%5.1f' %pi)
print('%05.1f' %pi)
print('%+f' %pi)
print('% f' %pi)
print('%-10f' %a)

data=('John','Doe',53.44)
format_string='Hello'
print('%s %s %s. Your current balance is $%.2f' %(format_string,data[0],data[1],data[2]))
print()
#list
a=[99,'bottles of beer',['on','the','wall']]
a*3
a[-1]
a[1:]
len(a)
a[0]=98
a[1:2]=['bottles','of','beer']
del a[-1]

a=list(range(5))
a.append(5)
a.pop()
a.insert(0,42)
a.pop(0)
a.reverse()
a.sort()
a

#Dictionaries
d={'duck':'eend','water':'water'}
d['duck']
del d['water']
d['back']='rug'
d['duck']='duik'
d

d.keys()
d.values()
d.items()
d.has_key('duck')
d.has_key('spam')
{'name':'Guido','age':43,('hello','world'):1,42:'yes','flag':['red','white','blue']}

#Tuples
lastname='Lee'
firstname='jae'
key=(lastname,firstname)
x=1
y=2
z=3
point=x,y,z
point
x,y,z=point
lastname=key[0]
lastname
singleton=(1,)
singleton
empty=()
empty

x='blue,red,green'
x.split(',')
a,b,c=x.split(',')
a
b
c

a=input('문장 입력 :')
list_a=a.split(' ')
list_a
list_b=[]
for i in (list_a):
    print(i.capitalize())
    list_b.append(i.capitalize())
list_b

number=input('숫자 입력:')
list_number=number.split(',')
tuple_number=tuple(list_number)
list_number
tuple_number

#if
#홀짝수 구분
inputNum=int(input('숫자를 입력하세요 :'))
if inputNum%2==0:
    print('짝수')
else:
    print('홀수')
    
inputNum=input('숫자를 입력하세요 :')
num=int(inputNum[-1])
num
if num==0 or num==2 or num==4 or num==6 or num==8:
    print('짝수')
else:
    print('홀수')

num=3.4
if num>0:
    print('Positive numnber')
elif num==0:
    print('Zero')
else:
    print('Negative number')
#성적으로 등급 매기기
inputGrade=int(input('성적을 입력해주세요 :'))
if inputGrade>90:
    print('The letter grade is A')
elif inputGrade>80:
    print('The letter grade is B')
else:
    print('The letter grade is C')
#3개의 숫자중 가장 큰수 찾기
inputNum=[]
for i in range(3):
    temp=int(input('숫자를 입력하세요:'))
    inputNum.append(temp)
inputNum
inputNum.sort()
print('The largest number between %d,%d and %d is %d' %(inputNum[0],inputNum[1],inputNum[2],inputNum[2]))

#계산기 
select=int(input('Select operation  \n 1. Add \n 2. Subtract \n 3. Multiply \n 4. Divide \n : '))
if select==1:
    inputNum1=int(input('Enter first number: '))
    inputNum2=int(input('Enter second number: '))
    print('%d + %d = %d' %(inputNum1,inputNum2,inputNum1+inputNum2))
elif select==2:
    inputNum1=int(input('Enter first number: '))
    inputNum2=int(input('Enter second number: '))
    print('%d - %d = %d' %(inputNum1,inputNum2,inputNum1-inputNum2))
elif select==3:
    inputNum1=int(input('Enter first number: '))
    inputNum2=int(input('Enter second number: '))
    print('%d * %d = %d' %(inputNum1,inputNum2,inputNum1*inputNum2))
elif select==4:
    inputNum1=int(input('Enter first number: '))
    inputNum2=int(input('Enter second number: '))
    print('%d / %d = %d' %(inputNum1,inputNum2,inputNum1/inputNum2))
else:
    print('잘못입력')

#가위바위보
player1=input('player1 가위/바위/보 중 선택 :')
player2=input('player2 가위/바위/보 중 선택 :')
if (player1=='가위' and player2=='보') or (player1=='바위' and player2 =='가위') or (player1=='보' and player2=='바위'):
    print('player1 Win!')
elif (player1=='가위' and player2=='바위') or (player1=='바위' and player2=='보') or (player1=='보' and player2=='가위'):
    print('player2 Win!')
elif (player1=='가위' and player2=='가위') or (player1=='바위' or player2=='바위') or (player1=='보' and player2=='보'):
    print('무승부!')
else:
    print('ERROR!')
    
