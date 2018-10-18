# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 14:36:50 2018

@author: SDEDU
"""

import pandas as pd
import numpy as np

data=pd.read_csv('brain_size.csv',sep=';',na_values='.')
data
t=np.linspace(-6,6,20)
sin_t=np.sin(t)
cos_t=np.cos(t)

pd.DataFrame({'t':t,'sin':sin_t,'cos':cos_t})

data.shape
data.columns
print(data['Gender'])
data[data['Gender']=='Female']['VIQ'].mean()

#group by
bygender=data.groupby('Gender')
for gender,value in bygender['VIQ']:
    print((gender,value.mean()))
bygender.mean()

#hypothesis testing
from scipy import stats
#t-test
#one sample test
stats.ttest_1samp(data['VIQ'],0)

#two sample test
female_viq=data[data['Gender']=='Female']['VIQ']
male_viq=data[data['Gender']=='Male']['VIQ']
stats.ttest_ind(female_viq,male_viq)

#linear model with category variables

#outlier detection
x=np.array([1,1,2,3,3,3,4,4,5,6])
y=np.array([45,55,50,75,40,45,30,35,25,15])
df=pd.DataFrame({'x':x,'y':y})
import seaborn as sns
sns.lmplot('y','x',data=df)

import statsmodels.api as sm
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()

y_pred=model.predict(x)
residual=y-y_pred
std_residual=residual/np.std(residual)
#standard residual plot
import matplotlib.pyplot as plt
plt.scatter(y_pred, std_residual)
plt.grid()

sns.residplot(x,y)

#outlier correction
df.y[3]=30
sns.lmplot('y','x',data=df)
x=df.x
y=df.y
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()

#regression calculation
y_pred=model.predict(x)
residual=y-y_pred
std_residual=residual/np.std(residual)

#standard residual plot (scatter, residplot)
plt.scatter(y_pred,std_residual)
sns.residplot(x,y)

#transformation
weight={'weight':[2289,2113,2180,2448,2026,2702,2657,2106,3226,3213,3607,2888],
        'mpg':[28.7,29.2,34.2,27.9,33.3,26.4,23.9,30.5,18.1,19.5,14.3,20.9]}
df=pd.DataFrame(weight)
plt.scatter(df.weight,df.mpg)
#regression model
import statsmodels.api as sm
x=df.weight
x=sm.add_constant(x)
model=sm.OLS(df.mpg,x).fit()
model.summary()
'''
1. Equation : mpg= -0.0116*weight +  56.0957
2. r sqaured : 0.935 -> Good
3. F-test : 2.85e-07 , 144.8
4. T-test : 0.000 ->  0.05 보다 작으므로 굳
'''
#residual plot
y_pred=model.predict(x)  #y_pred:예측값, df.mpg(=y) 값과 비교
residual=y - y_pred #잔차
std_residual = residual / np.std(residual) #표준 잔차
plt.scatter(y_pred,std_residual)
plt.grid()

#log tranformation of dependent variable(=y)
mpg_log=np.log(df.mpg)

#regression (weight --> Log_mpg)
x=df.weight
y=mpg_log
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()
'''
1. Equation : mpg_log = -0.0005*weight + 4.5242
2. r sqaured : 0.948
'''

#residual plot
y_pred=model.predict(x)
residual=y-y_pred
std_residual=residual/np.std(residual)
plt.scatter(y_pred,std_residual)
plt.grid()
sns.residplot(df.weight,df.mpg)

#reciprocal transformation
df.mpg_inverse=1/(df.mpg)

#regression
x=df.weight
y=df.mpg_inverse
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()
'''
1. Equation : mpg_inverse = 2.253e-05 * weight + -0.0173
2. r squared : 0.927
3. F-test : 5.32e-07, 126.7
4. T-test : 0.009
'''
#residual plot
y_pred=model.predict(x)
residual=y-y_pred
std_residual=residual/np.std(residual)
plt.scatter(y_pred,std_residual)
plt.grid()
sns.residplot(x,y)

#3#curvilinear regression
#reynolds example
data={'Months Employed':[41,106,76,104,22,12,85,111,40,51,9,12,6,56,19],
      'Scales Sold':[275,296,317,376,162,150,367,308,189,235,83,112,67,325,189]}
df=pd.DataFrame(data)

#regression model -> equation, r squared, f test, t test
x=df['Months Employed']
y=df['Scales Sold']
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()
'''
equation : sold = 2.3768*month + 111.2279
r squared : 0.781
f test : 1.24e-05 , 46.41
t test  : (const: 5.143, month: 6.812)
'''

#residual plot
y_pred=model.predict(x)
residual=y-y_pred
std_residual=residual/np.std(residual)
plt.scatter(y_pred,std_residual)
plt.grid()
sns.residplot(x,y)

#curvilinear model (y = ax^2 + bx + c)
df['month_sq']=df['Months Employed'] **2
#regression
x=df[['Months Employed','month_sq']]
y=df['Scales Sold']
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()
'''
equation : sold = -0.0345*(month_sq)+ 6.3448*month + 45.3476
'''
#residual plot
y_pred=model.predict(x)
residual=y-y_pred
std_residual=residual/np.std(residual)
plt.scatter(y_pred,std_residual)
plt.grid()
sns.residplot(x,y)
