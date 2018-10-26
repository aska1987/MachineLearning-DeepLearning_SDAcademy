# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 09:30:11 2018

@author: SDEDU
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#bicycle sale forecast
year = np.arange(1,11) 
sales = np.array([21.6, 22.9, 25.5, 21.9, 23.9, 27.5, 31.5, 29.7, 28.6, 31.4])
df = pd.DataFrame({'year':year,
              'sales':sales})
plt.plot(df.year, df.sales)
plt.ylim(0,35)
plt.title("Bicycle Sales Time Series")

import statsmodels.api as sm
x = df.year
y = df.sales
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
model.summary()

#forecasting equation
sales = 20.4000 + 1.1000*year

#forecast year 11 and year 12
year11 = 20.4000 + 1.1000*11
year12 = 20.4000 + 1.1000*12

plt.plot(df.year, df.sales)
plt.plot(df.year, model.predict(x))
plt.plot([10,11,12], [y[9], year11, year12])
plt.ylim(0,35)
plt.xlim(0,15)
plt.legend(['actual', 'predicted'])
plt.title("Bicycle Sales Time Series")

#forecast error
y_pred = model.predict(x)
error = y - y_pred
abs_error = np.abs(error)
per_error = (abs_error/y)*100
sq_error = error**2
me = error.mean()
mae = abs_error.mean()
mape = per_error.mean()
mse = sq_error.mean()
me,mae,mape,mse

cum_error = np.cumsum(error)
cum_abs_error = np.cumsum(abs_error)
mad = cum_abs_error/year
ts = cum_error / mad

plt.plot(year, ts)
plt.ylim(-5, 5)
plt.grid()
plt.title("Tracking Signals")

#choresterol sale forecast
revenue = np.array([23.1, 21.3, 27.4, 34.6, 33.8, 43.2, 59.5, 64.4, 74.2, 99.3])
year = np.arange(1,11)
df = pd.DataFrame({'year':year,
                   'rev':revenue})
plt.plot(df.year, df.rev)
plt.ylim(0,120)
df['year_squared'] = year**2

#quaratic regression
y = df.rev
x = df[['year', 'year_squared']]
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
model.summary()

#forecast equation
revenue = 24.1817 -2.1060 * year + 0.9216  * year **2

#forecast year 11 and year 12
rev11 = 24.1817 - (2.1060 * 11) + (0.9216  * (11 **2))
rev12 = 24.1817 - (2.1060 * 12) + (0.9216  * (12 **2))

#forecast error
y_pred = model.predict(x)
error = y - y_pred
abs_error = np.abs(error)
per_error = (abs_error/y)*100
sq_error = error**2
me = error.mean()
mae = abs_error.mean()
mape = per_error.mean()
mse = sq_error.mean()
me,mae,mape,mse

cum_error = np.cumsum(error)
cum_abs_error = np.cumsum(abs_error)
mad = cum_abs_error/year
ts = cum_error / mad

plt.plot(year, ts)
plt.ylim(-5, 5)
plt.grid()
plt.title("Tracking Signals")

#umbrella sale forecast
year = [1]*4+[2]*4+[3]*4+[4]*4+[5]*4
quarter = [1, 2, 3, 4]*5
sales = [125,153,106,88,118,161,133,102,138,144,113,80,109,
         137,125,109,130,165,128,96]
df = pd.DataFrame({'year':year,
             'qrt':quarter,
             'sales':sales},columns=['year', 'qrt', 'sales'])

#simple forecast with seasonality
qrt1 = df.sales[df.qrt==1].mean()
qrt2 = df.sales[df.qrt==2].mean()
qrt3 = df.sales[df.qrt==3].mean()
qrt4 = df.sales[df.qrt==4].mean()
qrt1, qrt2, qrt3, qrt4

#linear regression
df['qrt1'] = [1, 0, 0, 0] *5
df['qrt2'] = [0, 1, 0, 0] *5
df['qrt3'] = [0, 0, 1, 0] *5

y = df.sales
x = df[['qrt1', 'qrt2','qrt3']]

x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
model.summary()

sales = 95 + 29*qrt1 + 57*qrt2 + 26*qrt3
y6qrt1 = 95 + 29*1 + 57*0 + 26*0
y6qrt2 = 95 + 29*0 + 57*1 + 26*0
y6qrt3 = 95 + 29*0 + 57*0 + 26*1
y6qrt4 = 95 + 29*0 + 57*0 + 26*0
y6qrt1, y6qrt2, y6qrt3, y6qrt4

#seasonal forecasting with trend
year = [1]*4 + [2]*4 + [3]*4 + [4]*4
qrt = [1, 2, 3, 4]*4
sales = [4.8,4.1,6,6.5,5.8,5.2,6.8,7.4,6,5.6,7.5,7.8,6.3,5.9,8,8.4]
df = pd.DataFrame({'year':year,
                   'qrt':qrt,
                   'sales':sales}, columns=['year','qrt','sales'])
df['qrt1'] = [1, 0, 0, 0] *4
df['qrt2'] = [0, 1, 0, 0] *4
df['qrt3'] = [0, 0, 1, 0] *4
df['t'] = np.arange(1,17)

#regression
import statsmodel.api as sm
y = df.sales
x = df[['qrt1', 'qrt2', 'qrt3', 't']]
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
model.summary()
y_pred = model.predict(x)

sales = 6.0688 + -1.3631*qrt1 + -2.0337 *qrt2 + -0.3044* qrt3 +  0.1456 *t 
qrt1 = 6.0688 + -1.3631*1 + -2.0337 *0 + -0.3044* 0 +  0.1456 *17 
qrt2 = 6.0688 + -1.3631*0 + -2.0337 *1 + -0.3044* 0 +  0.1456 *18
qrt3 = 6.0688 + -1.3631*0 + -2.0337 *0 + -0.3044* 1 +  0.1456 *19  
qrt4 = 6.0688 + -1.3631*0 + -2.0337 *0 + -0.3044* 0 +  0.1456 *20  
qrt1, qrt2, qrt3, qrt4
plt.plot(sales)
plt.plot([16, 17,18,19,20],[y_pred[16],qrt1, qrt2, qrt3, qrt4])
plt.ylim(0,10)

