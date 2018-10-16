# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:46:06 2018

@author: SDEDU
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#butler example
miles=np.array([100,50,100,100,50,80,75,65,90,90])
time=np.array([9.3,4.8,8.9,6.5,4.2,6.2,7.4,6,7.6,6.1])
butler=pd.DataFrame()
butler=pd.DataFrame(miles,columns=['miles'])
butler['time']=time

plt.scatter(miles,time)
plt.xlabel('miles traveled')
plt.ylabel('travel time')

#Least squared method(최소제곱법)
x=miles
y=time
x.mean()
y.mean()

x-x.mean()
y-y.mean()

(x-x.mean())*(y-y.mean())
(x-x.mean())**2
a=(sum((x-x.mean())*(y-y.mean())))/sum((x-x.mean())**2)
round(y.mean()-round(a,4)*x.mean(),4) #intercept
'''
  y=ax+b
=>y=0.0698x+1.276
y=운전시간(time) x=운전거리(miles)
-> time=0.0698*miles+1.276 (prediction equation)
'''

import seaborn as sns
sns.lmplot('miles','time',data=butler)
sns.residplot(x,y)

#r squred value => r^2= SSR/SST
y_pred=0.0678*x+1.276
SSE=sum((y-y_pred)**2) #작을 수록 좋음
SST=sum((y-y.mean())**2) 
SSR=SST-SSE #클수록 좋음
rsquared=SSR/SST

#another variable
butler['delivery']=np.array([4,3,4,2,2,2,3,4,3,2])
x=butler[['miles','delivery']]
y=butler.time

import statsmodels.api as sm
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()
'''
model.summary()결과
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         -0.8687      0.952     -0.913      0.392      -3.119       1.381
miles          0.0611      0.010      6.182      0.000       0.038       0.085
delivery       0.9234      0.221      4.176      0.004       0.401       1.446


prediction equation : time = -0.8687 + 0.0611*miles + 0.9234*delivery
ex) miles=10, delivery=5 인 예측식은
  = -0.8687 + 0.0611*10 + 0.9234*5
'''

#miles to time
x=butler.miles
y=butler.time
x=sm.add_constant(x)
##model=sm.OLS('time~miles',data=butler).fit()
model=sm.OLS(y,x).fit()

#residual (잔차) = travel time - predicted travle time
y_pred=model.predict(x)
residual=y-y_pred

##scattplot
#miles to residual
plt.scatter(miles,residual)
plt.grid()
#delivery to residual
plt.scatter(butler.delivery,residual)
plt.grid()

#time to residual
plt.scatter(time,residual)
plt.grid()

#predicted time to residual
plt.scatter(y_pred,residual)

#standard residual
std_residual=residual/np.std(residual)

#standard residual plot(predicted time to std residual)
plt.scatter(y_pred,std_residual)

###regression using effort.csv
df=pd.read_csv('C:\\Users\\SDEDU\\Documents\\GitHub\\MachineLearning-DeepLearning_SDAcademy\\Python\\test\\effort.csv')
# x= effort , y= change
x=df.effort
y=df.change
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
y_pred=model.predict(x)
residual=y-y_pred
plt.scatter(x,residual)
plt.scatter(df.setting,residual)
plt.scatter(y_pred,residual)

std_residual=residual/np.std(residual)
plt.scatter(x,std_residual)


