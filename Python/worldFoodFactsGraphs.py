# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 17:40:52 2018

@author: SDEDU
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('C:\\Users\SDEDU\Downloads\en.openfoodfacts.org.products.tsv',delimiter='\t',usecols=[0,2,7,12,31])
da=pd.read_csv('C:\\Users\SDEDU\Downloads\en.openfoodfacts.org.products.tsv',delimiter='\t',nrows=1000,usecols=[0,2,7,12,31])

df_brand=data.brands.value_counts().head(10)
df_brand.plot.pie()
plt.title('brand')
plt.axis('equal')
plt.savefig('brand.pdf',format='pdf')
plt.show()

df_country=data.countries.value_counts().head(10)
df_country.plot.pie()
plt.title('country')
plt.axis('equal')
plt.savefig('country.pdf',format='pdf')
plt.show()

df_creator=data.creator.value_counts().head(10)
plt.xticks(rotation=30)
df_creator.plot(kind='barh')
plt.savefig('creator.pdf',format='pdf')

df_product_name=data.product_name.value_counts().head(10)
df_product_name.plot.barh()
plt.savefig('product_name.pdf',format='pdf')
#https://steemit.com/utopian-io/@tehshizno/pandas-visualization-1-exploring-nominal-data-using-bar-plots
conttab = pd.crosstab(data['creator'], data['countries'])
conttab

df_US_sub=data[data['countries']=='US']
df_US_sub.brands.value_counts().head(10).plot.pie()
plt.savefig('US_brand.pdf',format='pdf')

df_France_sub=data[data['countries']=='France']
df_France_sub.brands.value_counts().head(10).plot.pie()
plt.savefig('France_brand.pdf',format='pdf')