# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 16:46:37 2018

@author: SDEDU
"""

#seasonal decomposition exercise with us census data
#souvenior sales example
souvenior=pd.read_csv('https://robjhyndman.com/tsdldata/data/fancy.dat',
                      header=None)
souvenior.index=pd.date_range('1987-12','1994-12',freq='M')# 1987년 12월 부터 1994년 11월 까지
souvenior.columns=['sales']
ts=souvenior.sales
plt.plot(ts)
souvenior_log=np.log(ts) #승법 모델일때 로그변환 등을 시행 
plt.plot(souvenior_log)

from statsmodels.tsa.seasonal import seasonal_decompose
decom=seasonal_decompose(souvenior_log)
seasonal=decom.seasonal
souvenior_adj=souvenior_log-seasonal

t=np.arange(1,85)
import statsmodels.api as sm
t=sm.add_constant(t)
model=sm.OLS(souvenior_adj,t).fit()
model.summary()
souvenior_adj_trend=8.2636+0.0225*t
souvenior_adj_trend=model.predict(t)
souvnior_log_pred=souvenior_adj_trend+seasonal
souvnior_pred=np.exp(souvnior_log_pred)

plt.plot(ts)
plt.plot(souvenior_log)
plt.plot(souvenior_adj)
plt.plot(souvenior_adj_trend)
plt.plot(souvnior_log_pred)
plt.plot(souvnior_pred)

#forecast 1995 sales
t=np.arange(85,97)
souvenior_adj_trend=8.2636+0.0225*t
souvenior_adj_trend=model.predict(t)
souvnior_log_pred=souvenior_adj_trend+seasonal[0:12]
souvnior_pred=np.exp(souvnior_log_pred.values)

#gasoline data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sales=[17,21,19,23,18,16,20,18,22,20,15,22]
t=np.arange(1,13)
gas=pd.DataFrame({'t':t,'sales':sales},columns=['t','sales'])
plt.plot(gas.sales)
plt.ylim(0,25)
#simple moving average
gas_pred=[]
for i in range(len(sales)-2):
    gas_pred.append((sales[i+2]+sales[i+1]+sales[i])/3)
gas_pred
plt.plot(gas.sales)
plt.plot(np.arange(4,14),gas_pred)
plt.ylim(0,25)
plt.legend(['sales','prediction'])

#weighted moving average
#3WMA (1/6 , 2/6, 3/6)
gas_pred=[]
for i in range(len(sales)-2):
    gas_pred.append((sales[i+2]*(3/6)+sales[i+1]*(2/6)+sales[i]*(1/6)))
gas_pred
plt.plot(gas.sales)
plt.plot(np.arange(4,14),gas_pred)
plt.ylim(0,25)
plt.legend(['sales','3WMA'])

#exponential smoothing
gas_pred=[0]*13
gas_pred[1]=sales[0]
for i in range(len(sales)-1):
    gas_pred[i+2]=sales[i+1]*(.2)+gas_pred[i+1]*(.8) #알파는 0.2
gas_pred
plt.plot(gas.sales)
plt.plot(gas_pred)
plt.ylim(0,25)
plt.legend(['sales','exp. smoothing'])

#alpha= .3
gas_pred2=[0]*13
gas_pred2[1]=sales[0]
for i in range(len(sales)-1):
    gas_pred2[i+2]=sales[i+1]*(.3)+gas_pred2[i+1]*(.7) #알파는 0.2
gas_pred2
plt.plot(gas.sales)
plt.plot(gas_pred2)
plt.ylim(0,25)
plt.legend(['sales','exp. smoothing'])

#alph= .5
gas_pred3=[0]*13
gas_pred3[1]=sales[0]
for i in range(len(sales)-1):
    gas_pred3[i+2]=sales[i+1]*(.5)+gas_pred3[i+1]*(.5) #알파는 0.2
gas_pred3
plt.plot(gas.sales)
plt.plot(gas_pred3)
plt.ylim(0,25)
plt.legend(['sales','exp. smoothing'])

