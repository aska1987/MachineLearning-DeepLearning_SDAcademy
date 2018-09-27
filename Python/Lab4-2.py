# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 14:43:41 2018

@author: SDEDU
"""

#Inheritance
class Parent:
    parenAttr=100
    def __init__(self):
        print('Calling parent constructor')
    def parentMethod(self):
        print('Calling parent method')
    def setAttr(self,attr):
        Parent.parentAttr=attr
    def getAttr(self):
        print('Parent attribute :',Parent.parentAttr)

class Child(Parent):
    def __init__(self):
        print('Calling child constructor')
    def childMethod(self):
        print('Calling child method')

c=Child()
c.childMethod()
c.parentMethod()
c.setAttr(200)
c.getAttr()

class Employee:
    'Common base class for all employees'
    count=0
    def __init__(self,name,salary):
        self.name=name
        self.salary=salary
        self.count+=1
    def displayCount(self):
        print('total number of employees is ',self.count)
    def displayEmployee(self):
        print('name: ',self.name,'salary: ',self.salary)
        
e1=Employee('Zara',2000)
e1.name
e1.name='John'
setattr(e1,'name','setZara')
getattr(e1,'name')
hasattr(e1,'age')
delattr(e1,'age')
e1.displayCount()
e1.displayEmployee()

#built-in attributes
print(Employee.__doc__) #class documentation
print(Employee.__name__) #class name
print(Employee.__module__) #module name
print(Employee.__bases__) #base classes
print(Employee.__dict__) #dictionary
#pass Statement
for letter in 'Python':
    if letter=='h':
        pass

    print('This is pass block')
    print('Current Letter :', letter)
print('Good bye!')

#Define a class named American and its subclass NewYorker. You can use pass for null operations
class American:
    pass

class NewYorker(American):
    pass
a=American()
n=NewYorker()
print(a)
print(n)
#Define a class named Shape and its subclass Square.
#The Square class has an init function which takes a
#length as argument. Both classes have a area function
#which can print the area of the shape where Shape's
#area is 0 by default. You can use pass for null
#operations
class Shape:
    def printArea(self):
        print(self.area)
    def getArea(self):
        return 0
class Square(Shape):
    def __init__(self,length):
        self.length=length
    def getArea(self):
        print(self.length*self.length)
s=Square(3)
print(s.getArea())
#Define a class Person and its two child classes:
#Male and Female. All classes have a method
#"getGender" which can print "Male" for Male class
#and "Female" for Female class.
class Person:
    def getGender(self):
        pass
class Male(Person):
    def getGender(self):
        print('Male')
class Female(Person):
    def getGender(self):
        print('Female')
    
m=Male()
f=Female()
print(m.getGender())
print(f.getGender())

class Person:

    def __init__(self, first, last):
        self.firstname = first
        self.lastname = last

    def Name(self):
        return self.firstname + " " + self.lastname

class Employee(Person):

    def __init__(self, first, last, staffnum):
        Person.__init__(self,first, last)
        self.staffnumber = staffnum

    def GetEmployee(self):
        return self.Name() + ", " +  self.staffnumber

x = Person("Marge", "Simpson")
y = Employee("Homer", "Simpson", "1007")

print(x.Name())
print(y.GetEmployee())

#encapsulation
class JustCounter:
    __secretCount=0
    
    def count(self):
        self.__secretCount +=1
        print(self.__secretCount)
        
counter=JustCounter()
counter.count()
counter.count()
print(counter.__secretCount)
print(counter._JustCounter__secretCount)

#getter and setter
class MyClass:
    def setAge(self,num):
        self.age=num
    def getAge(self):
        return self.age
zack=MyClass()
zack.setAge(45)
print(zack.getAge())
zack.setAge('fourty five')
print(zack.getAge())

#private variable
class Circle:
    __radius=5
    def getRadius(self):
        return self.__radius
    def setRadius(self,num):
        self.__radius=num
    def getArea(self):
        return self.__radius **2 *3.14
    def getPerimeter(self):
        return self.__radius *2*3.14
c=Circle()
c.getArea()
c.__radius
c.getRadius()

#polymorphism
class Shark():
    def swim(self):
        print('The shark is swimming.')
    def swim_backwards(self):
        print('The shark cannot swim backwards, but can sink backwards.')
    def skeleton(self):
        print("The shark's skeleton is made of cartilage.")
class Clownfish():
    def swim(self):
        print('The clownfish is swimming.')
    def swim_backwards(self):
        print('The clownfish can swim backwards.')
    def skeleton(self):
        print("The clownfish's skeleton is made of bone.")  
sammy=Shark()
sammy.skeleton()
casey=Clownfish()
casey.skeleton()
sammy.swim_backwards()
casey.swim_backwards()

#Write two classes: Bear and Dog, both can make
#a distinct sound. You then make two instances
#and call their action using the same method.

class Bear:
    def makeSound(self):
        print('Groarrr')
class Dog:
    def makeSound(self):
        print('Woof woof!')
b=Bear()
b.makeSound()

#Write an class Car which holds the structure
#drive() and stop(). You can use pass for null
#operations. Then write two classes Sportscar and
#Truck, both are a form of Car.
class Car:     
    def __init__(self,name):
        self.name=name
    def drive(self):
         print(self.name,' is driving Car!')
    def stop(self):
        print(self.name,' is stopping Car!')
class Truck(Car):
    def __init__(self,name):
        self.name=name
    def drive(self):
        print(self.name,' is driving Truck!')
    def stop(self):
        print(self.name,' is stopping Truck!')
class Sportscar(Car):
    def __init__(self,name):
        self.name=name
    def drive(self):
        print(self.name,' is driving Sportscar!')
    def stop(self):
        print(self.name,' is stopping Sportscar!')
sport=Sportscar('Kim')
sport.drive()
