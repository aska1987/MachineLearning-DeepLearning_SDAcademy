# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 14:17:43 2018

@author: seedp
"""

import itertools
l1 = [1,2,3]
l2 = [4,5]
x = []

for combination in itertools.product(l1, l2):
	x.append(combination)

print (x)

numberList = [1, 2, 3]
strList = ['one', 'two', 'three']

# No iterables are passed
result = zip()

# Converting itertor to list
resultList = list(result)
print(resultList)

# Two iterables are passed
result = zip(numberList, strList)

# Converting itertor to set
resultSet = set(result)
print(resultSet)
