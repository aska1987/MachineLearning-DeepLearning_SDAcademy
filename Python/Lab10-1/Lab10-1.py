# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 09:45:53 2018

@author: SDEDU
"""

import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()

plt.plot([1,2,3,4],[1,4,9,16])

plt.plot([1,2,3,4],[1,4,9,16],'ro')
plt.axis([0,6,0,20])
plt.show()

import numpy as np
t=np.arange(0.,5.,0.2)
plt.plot(t,t,'r--',t,t**2,'bs',t,t**3,'g^')
plt.show()

data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

plt.scatter('a', 'b', c='c', s='d', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()

#using xticks
import calendar
plt.xticks(np.arange(5),('Tom','Dick','Harry','Sally','Sue'))
plt.xticks(np.arange(12),calendar.month_name[1:13],rotation=20)
plt.xticks(x,['Aaaaaddddda','Bbbbbbddddd','dddddcccc','Ddddddd'],rotation=90)
plt.show()

#subplot
names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]
plt.figure(1, figsize=(9, 3))
plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.show()
#savefig
x = np.arange(0,10)
y = x^2
plt.plot(x,y,'r>')
plt.xlabel("Time")
plt.ylabel("Distance")
plt.title("Graph Drawing")
plt.savefig('timevsdist.pdf',format='pdf')
#The location of the legend
x = np.arange(0,10)
y = x ^ 2
z = x ^ 3
t = x ^ 4
plt.plot(x,y)
plt.plot(x,z)
plt.plot(x,t)
plt.legend(['Race1','Race2','Race3'],loc=4)
#annotate
x = np.arange(0,10)
y = x ^ 2
plt.style.use('ggplot') #change style
#default figure  => import matplotlib as mpl
                    #mpl.rcParams.update(mpl.rcParamsDefault)
plt.plot(x,y)
plt.annotate(xy=[2,1], s='SecondEntry')
plt.annotate(xy=[4,6], s='Third Entry')

#3D surface plot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)
plt.show()

