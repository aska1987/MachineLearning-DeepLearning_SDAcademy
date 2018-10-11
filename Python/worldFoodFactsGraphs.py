# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 17:40:52 2018

@author: SDEDU
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('world-food-facts\en.openfoodfacts.org.products.tsv',delimiter='\t',usecols=[0,2,7,12,31])
da=pd.read_csv('world-food-facts\en.openfoodfacts.org.products.tsv',delimiter='\t',nrows=1000,usecols=[0,2,7,12,31])

df_brand=data.brands.value_counts().head(10)
df_brand.plot.pie()
plt.title('brand')
plt.axis('equal')
plt.show()

df_country=data.countries.value_counts().head(10)
df_country.plot.pie()
plt.title('country')
plt.axis('equal')
plt.show()

df_creator=data.creator.value_counts().head(10)
df_creator.plot(kind='barh')

df_product_name=data.product_name.value_counts().head(10)
df_product_name.plot.barh()

heat_data=[df_brand,df_country]
Index= ['I1', 'I2','I3','I4','I5']
Cols = ['C1', 'C2', 'C3','C4']
df_heat=pd.DataFrame(data,index=Index,columns=Cols)
plt.pcolor(df_heat.corr())
df_brand
#https://steemit.com/utopian-io/@tehshizno/pandas-visualization-1-exploring-nominal-data-using-bar-plots
conttab=pd.crosstab(data['brands'],data['countries'])
conttab
conttab.sort_values(by=)