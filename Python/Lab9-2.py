# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:58:23 2018

@author: SDEDU
"""

import matplotlib.pyplot as plt
plt.plot([1,2,3],[5,7,4])
plt.show()

x=[1,2,3]
y=[5,5,8]
x2=[1,2,3]
y2=[10,15,6]
plt.plot(x,y,label='first line')
plt.plot(x2,y2,label='second line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Line Chart')
plt.legend()
plt.show()
#scatter plot
x=[1,2,3,4,5,6,7,8]
y=[3,5,6,8,3,8,9,3]
plt.scatter(x,y,color='r',marker='x',s=100)
plt.xlabel('x')
plt.ylabel('y')
plt.title('scatter plot \n Example')
plt.show()

population_ages = [22, 55, 62, 45, 21, 22, 34, 42,
42, 4, 99, 102, 110, 120, 121, 122, 130, 111, 115,
112, 80,75, 65, 54, 44, 43, 42, 48]
bins = [0, 10, 20, 30, 40 , 50, 60, 70, 80, 90,
100, 110, 120, 130]
plt.hist(population_ages,bins,histtype='bar',rwidth=.8)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Histgram')
plt.show()

x=[1,2,3,4,5]
y=[6,7,8,2,4]
plt.bar(x,y,color='r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bar Chart')
plt.legend()
plt.show()

x=[1,2,3,4,5]
y=[6,7,8,2,4]
plt.barh(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bar Chart')
plt.show()
#bar chart
x=[2,4,6,8,10]
y=[6,7,8,2,4]
x2=[1,3,5,7,9]
y2=[7,8,2,4,2]
plt.bar(x,y,label='bars1',color='r')
plt.bar(x2,y2,label='bars2',color='c')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Tow Bar Charts')
plt.legend()
plt.show()
#stacked bar charts
x=[1,2,3,4,5]
y=[6,7,8,2,4]
y2=[7,8,2,4,2]
plt.bar(x,y,label='AAA',color='r')
plt.bar(x,y2,label='BBB',color='c',bottom=y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Stacked Bar Charts')
plt.legend()
plt.show()
#box plot
import numpy as np
import pandas as pd
x=np.random.randint(1,30,30)
plt.boxplot(x)
plt.xlabel('x')
plt.title('Box Plot')
plt.show()

data=[[1,2,3,4,5],
      [3,4,5,6,7],
      [6,7,7,8,8],
      [8,9,9,3,2]]
df=pd.DataFrame(data)
plt.boxplot(df)

#area chart
days = [1, 2, 3, 4, 5]
sleeping = [7, 8, 6, 11, 7]
eating = [2, 3, 4, 3, 2]
working = [7, 8, 7, 2, 2]
playing = [8, 5, 7, 8, 13]
plt.stackplot(days, sleeping, eating, working, playing, colors=['m', 'c','r', 'k'])
plt.plot([],[],color='m',label='sleeping', linewidth=5)
plt.plot([],[],color='c',label='eating', linewidth=5)
plt.plot([],[],color='r',label='working', linewidth=5)
plt.plot([],[],color='k',label='playing', linewidth=5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('interesting graph\n check it out')
plt.legend()
plt.show()

#heat map
data=[[2,3,4,1],[6,3,5,2],[6,3,5,4],[3,7,5,4],[2,8,1,5]]
Index=['I1','I2','I3','I4','I5']
Cols = ['C1', 'C2', 'C3','C4']
df=pd.DataFrame(data,index=Index,columns=Cols)
plt.pcolor(df.corr())
plt.colorbar()
plt.show()

#Pie chart
slices = [7, 2, 2, 13]
activities = ['sleeping','eating','working','playing']
cols = ['c','m','r','k']
plt.pie(slices,labels=activities,colors = cols,startangle=90,
        shadow=True,explode=(0, 0.1, 0, 0),autopct='%1.1f%%')
plt.title('Pie Chart')
plt.show()

