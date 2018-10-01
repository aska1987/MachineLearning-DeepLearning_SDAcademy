# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 09:54:29 2018

@author: SDEDU
"""

import numpy as np

a=np.array([1,2,3,4])
b=np.array([(1,2,3),(4,5,6)]) #2D array
c=np.array([[[1,2,3],[4,5,6]],[[3,2,1],[6,5,4]]]) #3D array
a.dtype #data type 
b.size #element 갯수
b.itemsize #element 하나가 차지하는 사이즈
b.shape #dimension
c.shape

np.zeros(3)
np.ones((3,4))
range(1,5)
np.arange(0,10,3)
np.linspace(0,100,6) # 6개의 숫자 생성 => 0, 20, 40, 60 ,80, 100 
np.arange(12).reshape(3,4) # 3행4열로 모양을 잡아준다

a=np.array([[1,2],[3,4],[5,6]])
a.ravel() # 1dimention으로 변경
a.flatten() # ravel 과 동일
a.T# transpose

#basic statistics
a.sum() # or np.sum(a)
a.sum(axis=0) #
a.sum(axis=1) #
a.min() # or np.min(a)
a.var() #평균
a.std() #표준편차
np.sqrt(a) # a.sqrt()는 안됨

a1=np.array([[1,2],[3,4]])
a2=np.array([[2,3],[4,5]])
a1.dot(a2) #행렬 곱

a=np.array([[5,6,7],[1,2,3],[7,5,3]])
a[1,2] # 3
a[0:2,2] # [7,3]
a[-1] # [7,5,3]
a[-1,0:2] #[7,5]
a[:,1:3] #[6,7],[2,3],[5,3]

#Iterating
a=np.array([[6,7,8],[1,2,3],[9,3,2]])

for row in a:
    print(row)
for cell in a.flat:
    print(cell)

#1. list [12.23,13.32,100,36.32] => 1d array
ex_list=[12.23,13.32,100,36.32]
np.array(ex_list)
#2. 3x3 matrix, from 2 to 10
np.array([[2,3,4],[5,6,7],[8,9,10]])
np.arange(2,11).reshape(3,3)
#3. null vector(size 10), 6th value => 11
ex_zeros=np.zeros(10)
ex_zeros[5]=11
ex_zeros
#4. 1d array, from 12 to 38
np.arange(12,39)
#5. [1,2,3,4] ==> float data type
ex_float=np.array([1,2,3,4,5],dtype='float64')
ex_float.dtype
#6. 5x5 matrix,border 1,inside=0
oz_matrix=np.zeros((5,5))
oz_matrix
for i in range(0,5):
    for j in range(0,5):
        if i==0 or i==4:
            oz_matrix[i,j]=1
        elif j==0 or j==4:
            oz_matrix[i,j]=1
            
q6=np.ones(25).reshape(5,5)
q6[1:4,1:4]=0
q6
#7. Celsius to Fahrenheit [0,12,45.21,34,99.91]
C_matrix=np.array([0,12,45.21,34,99.91])
C_matrix.itemsize
F_matrix=C_matrix*9+32.5    
F_matrix
#9. [1,2,3]--> number of elements, memory required for each elements, 
#   total memory required for all elements
N_matrix=np.array([1,2,3])
N_matrix.size
N_matrix.itemsize
N_matrix.dtype
N_matrix.size * N_matrix.itemsize

#stacking
a=np.arange(6).reshape(3,2)
a
b=np.arange(6,12).reshape(3,2)
b
np.vstack((a,b))
np.hstack((a,b))

#splitting
a=np.arange(30).reshape(2,15)
a
b=np.hsplit(a,3)
b[0]
b[1]
b[2]
c=np.vsplit(a,2)
c[0]
c[1]

#boolean arrays
a=np.arange(12).reshape(3,4)
b=a>4
a[b] # or a[a>4]
a[b]=-1
a
