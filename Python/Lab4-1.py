# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 09:34:01 2018

@author: SDEDU
"""
#module
import mymodule
mymodule.greeting('Lee')
mymodule.person1['age']

import platform
x=platform.system()
print(x)

x=dir(platform)
print(x)

#file I/O
open('demofile.txt')
f=open('demofile.txt','r')
print(f.read())

f=open('demofile.txt','r')
print(f.read(5))

f=open('demofile.txt','r')
print(f.readline())

f=open('demofile.txt','r')
for x in f:
    print(x)
    
f=open('demofile.txt','a') #append
f.write('Now the file has one more line!')
f=open('demofile.txt','r')
f.read()

f=open('demofile.txt','w') #overwrite 
f.write('Woops! I have deleted the content!')

f=open('myfile.txt','x') #create

import os
os.remove('demofile.txt') #delete file
os.rmdir('myfolder') #delete folder

#class
class MyClass:
    x=5

cla=MyClass()
cla.x

class Shark:
    def swim(self):
        print('The shartk is swimming.')
    def be_awesome(self):
        print('The shark is being awesome.')
s1=Shark() #개체화
s2=Shark()
s1.swim()
s1.be_awesome()
s2.swim()
s2.be_awesome()

#class with construction
class Person:
    def __init__(self,name,age):
        self.name=name
        self.age=age
    def myfunc(self):
        print('hello my name is ' + self.name)
pl=Person('Lee',24)
print(pl.name)
print(pl.age)
pl.myfunc()

class Shark:
    def __init__(self,name):
        self.name=name
    def swim(self):
        print(self.name+' is swimming.')
    def be_awesome(self):
        print(self.name +' is being awesome.')
sammy=Shark('Sammy')
sammy.swim()
sammy.be_awesome()

class Person:
  def __init__(mysillyobject, name, age):
    mysillyobject.name = name
    mysillyobject.age = age

  def myfunc(abc):
    print("Hello my name is " + abc.name)

p1 = Person("John", 36)
p1.myfunc()

p1.age = 40 #modify

del pl.age #delete
del pl

#Lab 4-1
#Write a class named Korean which has a method called printNationality.
class Korean:
    def __init__(self,name,nationality):
        self.name=name
        self.nationality=nationality
    def printNationality(info):
        print(info.name + ' is from ' +info.nationality)
cont=Korean('Lee','Korea')
cont.printNationality()

#Write a class named Circle constructed by a radius and two methods which will compute the area and the perimeter of a circle
class Circle:
    def __init__(self,radius):
        self.radius=radius
    def area(info):
        print('radius: ', info.radius,' area: ',((info.radius)*(info.radius)),'pi')
    def perimeter(info):
        print('radius: ', info.radius,' perimeter: ',((info.radius*2)),'pi')
cir=Circle(5)
cir.area()
cir.perimeter()

#Write a class named Rectangle constructed by a length and width and a method which will compute the area of a rectangle.
class Rectangle:
    def __init__(self,length,width):
        self.length=length
        self.width=width
    def area(info):
        area=info.length*info.width
        print(info.length,' * ',info.width,' = ',area )
rec=Rectangle(5,4)
rec.area()

#Write a class which has two methods get_String and print_String. get_String accept a string from the user and print_String print the string in upper case.
class upperPrint:
    def __init__(self):
        self.UpInfo=""
    def get_String(self,string):
        self.UpInfo=(string).upper()        
    def print_String(self):
        print(self.UpInfo)
upP=upperPrint()
upP.get_String('abcdef')
upP.print_String()
#Write a class to reverse a string word by word. Input string : 'hello .py' Expected Output : '.py hello'
class reverseString:
    def __init__(self,info):
        self.info=info
    def output(self):
        for i in range(-1,-len(self.info),-1):
            print(self.info[i],end='')
            
rS=reverseString('hello.py')
rS.output()

