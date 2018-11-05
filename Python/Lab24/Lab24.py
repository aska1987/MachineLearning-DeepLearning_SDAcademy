# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 11:06:29 2018

@author: SDEDU
"""

import math
pi=math.pi
(1/10)*(math.sqrt((2*pi)))*(math.exp((-((1.96)**2)/2)+1))
1/(math.sqrt((2*pi)))*(math.exp((-((1.96)**2)/2)+1))

1/math.sqrt(2*math.pi)*math.exp(-((1.96**2)/2)+1)

my={'bird':'새'}
my['cat']='고양이'
my['cat']
my['고양이']

dic={'1':1,'2':2,'3':3,'4':4,'5':5,'6':6}
dic['1']+dic['6']==7 and dic['2']+dic['5']==7 and dic['3']+dic['4']==7

def max_min(a):
    print(max(a))
    print(min(a))
my_list=[1,3,2,9,6,5,3]
max_min(my_list)
import matplotlib.pyplot as plt
for i in range(0,6,0.1):
    print(i)
    plt(-(i-1)(i-3)(i-4))

x=[a for a in range(0,6)]
y=[(-(b-1)(b-3)(b-4)) for b in range(6)]
y=[a*a for a in x]
plt.plot(x,y)


import numpy as np
x=np.arange(-1,6,0.1)
y=-(x-1)*(x-3)*(x-4)
plt.plot(x,y)


x=np.arange(-1,11,0.1)
y=-(x-1)*(x-6)*(x-9)
y2=-(x-2)*(x-4)*(x-7)
plt.plot(x,y)
plt.plot(x,y2)
plt.legend()


from matplotlib.image import imreadimport imread
img=imread('lena.jpg')
plt.imshow(img)
plt.show()

#and gate
def AND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5, 0.5])
    b=-0.7
    tmp=np.sum(w*x)+b
    if tmp<=0:
        return 0
    else:
        return 1
if __name__=='__main__':
    for xs in [(0,0),(1,0),(0,1),(1,1)]:
        y=AND(xs[0],xs[1])
        print(str(xs)+ '->' + str(y))

        
#or gate
def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5, 0.5])
    b=-0.2
    tmp=np.sum(w*x)+b
    if tmp<=0:
        return 0
    else:
        return 1
if __name__=='__main__':
    for xs in [(0,0),(1,0),(0,1),(1,1)]:
        y=OR(xs[0],xs[1])
        print(str(xs)+ '->' + str(y))


#nand gate
def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5, 0.5])
    b=-0.7
    tmp=np.sum(w*x)+b
    if tmp>0:
        return 0
    else:
        return 1
if __name__=='__main__':
    for xs in [(0,0),(1,0),(0,1),(1,1)]:
        y=NAND(xs[0],xs[1])
        print(str(xs)+ '->' + str(y))




#xor gate (OR and NAND)

if __name__=='__main__':
    for xs in [(0,0),(1,0),(0,1),(1,1)]:
        y=(NAND(xs[0],xs[1])) and (OR(xs[0],xs[1]))
        print(str(xs)+ '->' + str(y))


def XOR(x1,x2):
    S1=AND(not(x1),x2)
    S2=AND(x1,not(x2))
    Y=OR(S1,S2)
    return Y

for i in [0,1]:
    for j in [0,1]:
        print(i,j,'->',XOR(i,j))

##