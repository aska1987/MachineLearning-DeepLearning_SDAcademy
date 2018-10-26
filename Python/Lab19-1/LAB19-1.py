import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.date_range('1/1/2018', periods=5, freq='M')
pd.date_range('1/1/2018', periods=5, freq='W')
pd.date_range('1/1/2018', periods=5, freq='D')
pd.date_range('1/1/2011', periods=5, freq='B')
pd.date_range('1/1/2018','5/5/2018')

dt = pd.date_range('2010-01', periods = 20, freq='Q')
sales = [125,153,106,88,118,161,133,102,138,144,113,80,109,
         137,125,109,130,165,128,96]
df = pd.DataFrame({'dt':dt, 'sales':sales},index=dt)
ts = df['sales']
ts['2010']
plt.plot(ts)
plt.ylim(0,200)
plt.title('Umbrella Time Series')

#shutterfly example
sfly = pd.read_csv("SFLY.csv", index_col='Date', 
                   parse_dates=['Date'])
sfly.dtypes
sfly.set_index("Date")
sfly.to_date('Date') #or sfly.to_datetime('Date')
sfly['2015-11'].mean()

ts = sfly['Close']
plt.plot(ts)
plt.xticks(rotation=60)

import statsmodels.api as sm
x = np.arange(1,len(ts)+1)
y = ts
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
model.summary()

price = 36.0230 + 0.0514 * x

#forecast next 30 days
newdate = np.arange(758, 758+31)
pred_price = 36.0230 + 0.0514 * newdate

#forecast chart
plt.plot(ts.values)
plt.plot(36.0230 + 0.0514 * np.arange(1,758+200))
plt.xticks(rotation=60)
plt.title('Shutterfly Stock Price Forecast')

#seasonal forecast using seasonal_decompose
dt = pd.date_range('2010-01', periods = 20, freq='Q')
sales = [125,153,106,88,118,161,133,102,138,144,113,80,109,
         137,125,109,130,165,128,96]
df = pd.DataFrame({'dt':dt, 'sales':sales},index=dt)
ts = df['sales']

from statsmodels.tsa.seasonal import seasonal_decompose
decom = seasonal_decompose(ts)
trend = decom.trend
seasonal = decom.seasonal
residual = decom.resid
plt.subplot(411)
plt.plot(ts)
plt.legend(['original'])
plt.subplot(412)
plt.plot(trend)
plt.legend(['trend'])
plt.subplot(413)
plt.plot(seasonal)
plt.legend(['seasonal'])
plt.subplot(414)
plt.plot(residual)
plt.legend(['residual'])
plt.tight_layout()

ts_adj = ts - seasonal
plt.plot(ts)
plt.plot(ts_adj)
plt.ylim(0,170)

#regression
y = ts_adj
x = np.arange(1,len(ts)+1)

import statsmodels.api as sm
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
model.summary()


ts_adj_trend = 119.3771 + 0.3450*x
ts_adj_trend = model.predict(x)
ts_pred = ts_adj_trend + seasonal

plt.plot(ts.values, label='sales')
plt.plot(ts_adj.values, label='sale adj.')
plt.plot(ts_adj_trend, label='adj. trend')
plt.plot(ts_pred.values, label='sales prediction')
plt.legend()
plt.ylim(0,200)

plt.plot(ts.values, label='sales')
plt.plot(ts_pred.values, label='sales prediction')
plt.legend()
plt.ylim(0,200)
#forecast year 6
ts_adj_trend = 119.3771 + 0.3450* np.arange(len(ts_adj)+1, 
                                           len(ts_adj)+5)
ts_pred = ts_adj_trend + seasonal.values[0:4]

#tv set forecast using seasonal decomposition
dt = pd.date_range('2010-01', periods = 16, freq='Q')
sales = [4.8,4.1,6,6.5,5.8,5.2,6.8,7.4,6,5.6,7.5,7.8,6.3,5.9,8,8.4]
df = pd.DataFrame({'dt':dt,
                   'sales':sales}, index=dt)
ts = df['sales']
plt.plot(ts)
plt.ylim(0,10)

#decompose
from statsmodels.tsa.seasonal import seasonal_decompose
decom = seasonal_decompose(ts)
trend = decom.trend
seasonal = decom.seasonal
residual = decom.resid

ts_adj = ts - seasonal
y = ts_adj
x = np.arange(1, len(ts_adj)+1)
import statsmodels.api as sm
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
model.summary()

ts_adj_trend = 5.1392 + 0.1461 * np.arange(1, len(ts_adj)+1)
ts_pred = ts_adj_trend + seasonal

#forecast chart
plt.plot(ts.values, label='sales')
plt.plot(ts_adj.values, label='sale adj.')
plt.plot(ts_adj_trend, label='adj. trend')
plt.plot(ts_pred.values, label='sales prediction')
plt.legend()
plt.ylim(0,12)

plt.plot(ts.values, label='sales')
plt.plot(ts_pred.values, label='sales prediction')
plt.legend()
plt.ylim(0,12)

#forecast year 5
ts_adj_trend = 5.1392 + 0.1461 * np.arange(len(ts_adj)+1, 
                                           len(ts_adj)+5)
ts_pred = ts_adj_trend + seasonal.values[0:4]

#clothing store 
clothing = pd.read_excel('clothing.xls'12 2arse_dates=['Period'], index_col='Period')
ts = clothing.Value
plt.plot(ts)
plt.ylim(0,600000)

decom = seasonal_decompose(ts)
seasonal = decom.seasonal
ts_adj = ts - seasonal
x = np.arange(1,len(ts_adj)+1)
y = ts_adj
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
model.summary()
ts_adj_trend = model.predict(x)
ts_pred = ts_adj_trend + seasonal

#
#forecast chart
plt.plot(ts.values, label='sales')
plt.plot(ts_adj.values, label='sale adj.')
plt.plot(ts_adj_trend, label='adj. trend')
plt.plot(ts_pred.values, label='sales prediction')
plt.legend()

plt.plot(ts.values, label='sales')
plt.plot(ts_pred.values, label='sales prediction')
plt.legend()

#forecast 2018 sales
x
newx=np.arange(61,73)
newx=sm.add_constant(newx)
ts_adj_trend=model.predict(newx)
yr_2018=ts_adj_trend+seasonal[0:12].values
yr_2018
plt.plot(ts.values,label='orignal')
plt.plot(ts_pred.values,label='predicted ales')
plt.plot(newx[:,1],yr_2018)
plt.legend()

#decompose (-seasonaity)-trend line - (+seasonality)
drinking = pd.read_excel('SeriesReport-201810260241.xls', 
                         parse_dates=['Period'], index_col='Period')
drinking=drinking.dropna()
ts=drinking.Value
plt.plot(ts)

decom=seasonal_decompose(ts)
seasonal=decom.seasonal
ts_adj=ts-seasonal
x=np.arange(1,len(ts_adj)+1)
y=ts_adj
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()
ts_adj_trend=model.predict(x)
ts_pred= ts_adj_trend + seasonal

#plot
plt.plot(ts.values, label='sales')
plt.plot(ts_adj.values, label='sale adj.')
plt.plot(ts_adj_trend, label='adj. trend')
plt.plot(ts_pred.values, label='sales prediction')
plt.legend()

plt.plot(ts.values, label='sales')
plt.plot(ts_pred.values, label='sales prediction')
plt.legend()

#seasonal decomposition exercise with us census data
#souvenior sales example
souvenior=pd.read_csv('https://robjhyndman.com/tsdldata/data/fancy.dat',
                      header=None)
souvenior.index=pd.date_range('1987-12','1994-12',freq='M')# 1987년 12월 부터 1994년 11월 까지
souvenior.columns=['sales']
ts=souvenior.sales
plt.plot(ts)
souvenior_log=np.log(ts)
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
