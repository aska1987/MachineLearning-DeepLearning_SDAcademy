# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 09:53:13 2018

@author: SDEDU
"""

import pandas as pd
import random

data = pd.read_csv('https://s3-eu-west-1.amazonaws.com/shanebucket/downloads/uk-500.csv')

data.head()
data.dtypes
data.shape
data.columns
data.ndim
data.size
data.values
data.info()
data.count()

data['id']=[random.randint(0,1000) for x in range(data.shape[0])]
data.head()
#iloc
data.iloc[0]
data.iloc[-1]
data.iloc[:,0]
data.iloc[:,-1]
data.iloc[0:5]
data.iloc[:,0:2]
data.iloc[[0,3,6,24],[0,5,6]]
data.iloc[0:5,5:8]

print(type(data.iloc[100])) # only row selected =>series
print(type(data.iloc[[100]])) #list selection used => dataframe

print(type(data.iloc[2:10])) #two rows selected => dataframe
print(type(data.iloc[1:2,3])) #only one column selected => Series
print(type(data.iloc[1:2,[3]])) #one column be only one column selected => dataframe

data.set_index('last_name',inplace=True)
data.head()
#loc
data.loc['Andrade']
data.loc[['Andrade','Veness']]
data.loc[['Andrade','Veness'],['first_name','address','city']]
data.loc['Andrade':'Veness',['first_name','address','city']]

#filtering
data={'name':['Kim','Park','Lee','Moon'],
      'salary':[1000,2000,3000,4000],
      'gender':['F','M','F','M'],
      'dept':['IT','Operations','Finance','Acconting']}
df=pd.DataFrame(data)
highsal=df[df.salary>2500]
female=df[df.gender=='F']
twodept=df[df.dept.isin(['IT','Operations'])]

#renaming
df.columns=['Name','Salary','Gender','Dept']
df.rename(columns={'Name':'name'},inplace=True)

#Filter the data and display the following
#columns and rows:
data=pd.read_csv('weather.txt')
data.head

#avg_precipitation > 1.0
data[data.avg_precipitation>1.0]
#Month is in either June, July, or August
data[data.month.isin(['Jun','Jul','Aug'])]

#Assign new values in the following locations:

#101.3 for avg_precipitation column at index 9
data.iloc[9,:]
 #data[data.avg_precipitation[9]]
 #data[data['avg_precipitation'][9]]=101.3
 #data[data.avg_precipitation[9]]=101.3
 #data.iloc[9,5]=101.3
data.loc[9,'avg_precipitation']=101.3
#np null values (np.nan) for avg_precipitation column at index 9
import numpy as np
 #data[data['avg_precipitation'][9]]=np.nan
data.loc[9,'avg_precipitation']=np.nan
data.iloc[9,:]
 #data.iloc[9,5]=np.nan
#5 for all rows in avg_low column
data.loc[:,'avg_low']=5
#Add new column named avg_day that is the average of avg_low and avg_high
data['avg_day']=(data.avg_low+data.avg_high)/2

#Rename columns
#avg_precipitation to avg_rain
data.rename(columns={'avg_precipitation':'avg_rain'},inplace=True)
#Change columns’ name to
#'month','av_hi','av_lo','rec_hi','rec_lo','av_rain','av_day‘
data.columns=['month','av_hi','av_lo','rec_hi','rec_lo','av_rain','av_day']

#Save the result data frame to a csv file
data.to_csv('weather_result.csv')




import pandas as pd
food=pd.read_csv("world-food-facts\en.openfoodfacts.org.products.tsv",nrows = 1000,delimiter='\t')
#Display the first 5 rows
food.head()
#What is the number of observations in the dataset?
len(food)
#What is the number of columns in the dataset?
len(food.columns)
#Print the name of all the columns.
food.columns

#What is the name of 105th column?
food.columns[104:105]

#What is the data type of the observations of the 105th column?
food.dtypes[104:105]
food.dtypes[105]
type(food.iloc[[105]])
#How is the dataset indexed?
food.index
#What is the product name of the 19th observation?
food.loc[18,'product_name']


#filtering
creator=food[food['creator'] == 'tacinte']

