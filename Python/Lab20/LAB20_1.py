import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = 15,6
#plt.figure(figsize=(15,6))

df = pd.read_csv('AirPassengers.csv')
df.head()
df.dtypes

import time
#dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
df = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month')
#date_parser=dateparse
df.dtypes
df.index
df.columns

ts = df['#Passengers']
ts.head(10)
ts['1949-01-01'] #specific date
ts['1949-01']  #specific month
ts['1949'] #specific year
ts['1949-01-01':'1949-05-01'] #period  
ts[:'1949-12-01'] #period
plt.plot(ts)      

#log transformation
ts_log = np.log(ts)
plt.plot(ts_log)

#seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
seasonal = seasonal_decompose(ts_log).seasonal
ts_adj = ts_log - seasonal
plt.plot(ts_adj)

#moving average
ts_ma = ts_adj.rolling(window=3).mean() # 3 month moving average
plt.plot(ts_ma)

#exponentially weighted moving average
ts_ewma = pd.ewma(ts_adj, halflife=12)

import statsmodels.api as sm
x = np.arange(1,len(ts_adj)+1)
y = ts_adj.values
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
model.summary()

#king example
f = open("https://robjhyndman.com/tsdldata/hurst/precip1.dat", "rb")
f.readlines()

#trend equation 
ts_adj_trend = 4.8126 + 0.0101 * x
ts_adj_trend = model.predict(x)
ts_log_pred = ts_adj_trend + seasonal
ts_pred = np.exp(ts_log_pred)

#forecast chart
plt.plot(ts, label='original')
plt.plot(ts_pred, label='predicted')
plt.legend()

#forecast month 145-155
newx = np.arange(145, 157)
newx = sm.add_constant(newx)        
ts_adj_trend = model.predict(newx)
ts_log_pred = ts_adj_trend + seasonal[0:12]
ts_pred = np.exp(ts_log_pred)        
ts_pred

#PPT
#king example
#birth example
#rainfall example
#skirt example
#volcano example


