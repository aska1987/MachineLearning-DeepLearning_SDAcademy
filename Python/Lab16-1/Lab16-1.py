# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 09:39:32 2018

@author: SDEDU
"""

import seaborn as sns;
import matplotlib.pyplot as plt

sns.set(style='ticks',color_codes=True)
tips=sns.load_dataset('tips')
g=sns.FacetGrid(tips,col='time',row='smoker')
g=g.map(plt.hist,'total_bill')

import numpy as np
bins=np.arange(0,65,5) #최소 숫자보다는 작게, 최대 숫자보다는 크게 생성해야한다
tips.total_bill.max() #50.81
tips.total_bill.min() #3.07
g=sns.FacetGrid(tips,col='time',row='smoker')
g=g.map(plt.hist,'total_bill',bins=bins,color='r')

g=sns.FacetGrid(tips,col='time',row='smoker')
g=g.map(plt.scatter,'total_bill','tip',edgecolor='w')

#scatter plot with legend
g=sns.FacetGrid(tips,col='time',hue='smoker')
g=(g.map(plt.scatter,'total_bill','tip',edgecolor='w').add_legend())

#aspect ratio
g=sns.FacetGrid(tips,col='day',aspect=.5)
g=g.map(plt.hist,'total_bill',bins=bins)


g = sns.FacetGrid(tips, col="smoker", col_order=["Yes", "No"])
g = g.map(plt.hist, "total_bill", bins=bins, color="m")

#custom palette with different makers
kws = dict(s=50, linewidth=.5, edgecolor="w")
g = sns.FacetGrid(tips, col="sex", hue="time", palette="Set1",
                  hue_order=["Dinner", "Lunch"])
g = (g.map(plt.scatter, "total_bill", "tip", **kws)
     .add_legend())

#col_wrap
att=sns.load_dataset('attention')
g=sns.FacetGrid(att,col='subject',col_wrap=5)
g=g.map(plt.plot,'solutions','score',marker='.')

#axis_labels
g=sns.FacetGrid(tips,col='smoker',row='sex')
g=g.map(plt.scatter,'total_bill','tip',color='g',**kws)
g=g.set_axis_labels('Total bill (US Dollars)','Tip')
g=g.set(xlim=(0,60),ylim=(0,12),xticks=[10,30,50],
        yticks=[2,6,10])


g=sns.FacetGrid(tips,col='size',col_wrap=3)
g=g.map(plt.hist,'tip',bins=np.arange(0,13),color='c')
g=g.set_titles('{col_name} dinners')
tips.tip.min() #1.0
tips.tip.max() #10.0


#geographical map tutorial
#https://scitools.org.uk/cartopy/docs/v0.15/matplotlib/intro.html
#conda install cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

ax=plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
plt.show()

ax=plt.axes(projection=ccrs.Mollweide())
ax.stock_img()
plt.show()

#Adding data to the map
ax = plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()
ny_lon, ny_lat = -75, 43
delhi_lon, delhi_lat = 77.23, 28.61

plt.plot([ny_lon, delhi_lon], [ny_lat, delhi_lat],
         color='blue', linewidth=2, marker='o',
         transform=ccrs.Geodetic(),
         )
plt.plot([ny_lon, delhi_lon], [ny_lat, delhi_lat],
         color='gray', linestyle='--',
         transform=ccrs.PlateCarree(),
         )
plt.text(ny_lon - 3, ny_lat - 12, 'New York',
         horizontalalignment='right',
         transform=ccrs.Geodetic())
plt.text(delhi_lon + 3, delhi_lat - 12, 'Delhi',
         horizontalalignment='left',
         transform=ccrs.Geodetic())
plt.show()

#tyler example
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_excel('tyler.xlsx')
df['PriceAdv']=df.Price * df.AdvExp

import statsmodels.api as sm
y=df.Sales
x=df.drop(['Sales'],axis='columns')
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()
'''
1. 예측식 : Sales =  -275.8333 + 175*Price + 19.68*AdvExp + -6.08*PriceAdv
2. r squared value : 0.978
3. F test : p value < 0.05  => 통계적으로 유의
4. T test : p value < 0.05  => 통계적으로 유의
'''

#residual analysis
y_pred = model.predict(x)
residual=y-y_pred
std_residual=residual/np.std(residual)
plt.scatter(y_pred,std_residual)
plt.grid()

#pivot table 
table=pd.pivot_table(df,values='Sales',
                     index=['AdvExp'],
                     columns=['Price'],
                     aggfunc=np.mean)

