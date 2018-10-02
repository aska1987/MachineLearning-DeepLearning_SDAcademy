# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 09:49:49 2018

@author: SDEDU
"""

import numpy as np
x=[0,1,2,3,4,5]
a=np.array(x)
a[2]
a[1:4:2]
a[3:]
a[:3]
a.shape
a.size
a.itemsize
a.dtype

b=np.array([[1,2,3],[4,5,6]])
b.swapaxes(0,1)

a=np.arange(0,6)
a
a=np.arange(0,6).reshape(2,3)
a

a=np.array([2,3,4])
a=np.arange(1,12,2)
a
a=np.linspace(1,12,6)
a

a.reshape(3,2)
a=a.reshape(3,2)
a.size
a.shape
a.dtype
a.itemsize

b=np.array([(1.5,2,3),(4,5,6)])
b

a<4
a*3
a*=3
a=np.zeros((3,4))
a
a=np.array([2,3,4],dtype=np.int16)
a
a=np.random.random((2,3))
a

np.set_printoptions(precision=2,suppress=True)
a=np.random.randint(0,10,5)
a
a.sum()
a.min()
a.max()
a.mean()
a.var()
a.std()

a.sum(axis=1)
a.sum(axis=0)

a.argmin()
a.argmax() #index of max element
a.argsort()
a.sort()

a=np.arange(10)**2
a
for i in a:
    print(i**2)
a[::-1] #reverse array

for i in a.flat:
    print(i)
a.transpose()
a.ravel() #1d로

data=np.loadtxt('data_file.txt',dtype=np.uint8,delimiter=',',skiprows=1,usecols=[0,1,2,3])
data

np.random.shuffle(a)
a

np.random.choice(a)
