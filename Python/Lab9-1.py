# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:42:07 2018

@author: SDEDU
"""

import pandas as pd
import numpy as np

raw_data={'first_name': ['Jason', np.nan, 'Tina', 'Jake', 'Amy'], 
        'last_name': ['Miller', np.nan, 'Ali', 'Milner', 'Cooze'], 
        'age': [42, np.nan, 36, 24, 73], 
        'sex': ['m', np.nan, 'f', 'm', 'f'], 
        'preTestScore': [4, np.nan, np.nan, 2, 3],
        'postTestScore': [25, np.nan, np.nan, 62, 70]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'sex', 'preTestScore', 'postTestScore'])
df.head()

#Drop missing observations
df_no_missing = df.dropna()
df_no_missing

#Drop rows where all cells in that row is NA
df_cleaned = df.dropna(how='all')
df_cleaned

#Create a new column full of missing values
df['location'] = np.nan
df

#Drop column if they only contain missing values
df.dropna(axis=1, how='all')
#Drop rows that contain less than five observations
df.dropna(thresh=5)
#Fill in missing data with zeros
df.fillna(0)

#Fill in missing in preTestScore with the mean value of preTestScore
df["preTestScore"].fillna(df["preTestScore"].mean(), inplace=True)
df

#Fill in missing in postTestScore with each sex’s mean value of postTestScore
df["postTestScore"].fillna(df.groupby("sex")["postTestScore"].transform("mean"), inplace=True)
df

#Select some raws but ignore the missing data points
df[df['age'].notnull() & df['sex'].notnull()]


#replacing data
df = pd.DataFrame({
    'name':['john','mary','paul'],
    'age':[30,25,40],
    'city':['new york','los angeles','london']
})

df.replace([25],40)
#multiple
df.replace({
    25:26,
    'john':'johnny'
})
#regex
df.replace('jo.+','FOO',regex=True)
#single column
df.replace({'age':{30:31}})



#Read the world food bank tsv file and assign it
#to a dataframe called food and complete the following tasks
data=pd.read_csv('world-food-facts\en.openfoodfacts.org.products.tsv',delimiter='\t')

#Check missing values, count them by columns, and count the total number of missing
data.isnull().sum()
(data.isnull().sum()).sum()
#Drop all missing observations
dataDrop=data.dropna(how='all')
#Drop columns where all cells in that column is NA
dataDrop=data.dropna(how='all',axis=1)
dataDrop
#Fill NA with the means of each column
dataDrop.replace(np.nan,dataDrop.mean(axis=1))
dataDrop.fillna(dataDrop.mean())

#Import the dataset and assign it to a variable called users and use the 'user_id' as index
data=pd.read_csv('Occupation.txt',delimiter='|')
#Display the first 25 rows
data.head(25)
#Display the last 10 rows
data.tail(10)
#What is the number of observations in the dataset?
data.size
#What is the number of columns in the dataset?
data.shape[1]

#Print the name of all the columns.
data.columns
#How is the dataset indexed?
data.index
#What is the data type of each column?
data.dtypes

#Print only the occupation column
data.loc[:,'occupation']
data.occupation
data['occupation']
#how many different occupations there are in this dataset?
len(data.occupation.value_counts())
#What is the most frequent occupation?
 data.occupation.value_counts().head(1)

#Summarize the dataframe (descriptive statistics)
data.describe()
#Summarize all the columns
data.describe(include='all')
#Summarize only the occupation column
data.occupation.describe()
#What is the mean age of users?
data.age.mean()
#What is the age with least occurrence?
a=data.age.value_counts()
a
a[a==np.min(a)]
