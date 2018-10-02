# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:32:35 2018

@author: SDEDU
"""

#Load data_file.txt using genfromtxt and create a 1d
#array for time (0 column) and a 2d array for sensor
#data (2-5 columns)
import numpy as np
array_file=np.genfromtxt('data_file.txt',delimiter=',')

#Display the first 6 sensor rows
array_file[0,]
#Adjust time to start at zero by subtracting the first element in the time vector (index = 0)
array_1d=array_file[:,0]
array_1d-array_1d[0]
#Calculate the average of the sensor readings
array_2d=array_file[:,1:5]
array_var=np.mean(array_2d,axis=1)
len(array_var)
len(array_1d)
len(array_2d)

array_var
#Stack time, sensor data, and avg as column vectors and ranspose data
array_1d=array_1d.reshape(1200,1)
array_var=array_var.reshape(1200,1)
result_array=np.hstack([array_1d,array_2d,array_var])
#Save text file with comma delimiter
np.savetxt('data_file_result.txt',result_array,delimiter=',')


#Data set
#BodyTemperature.txt includes data of a sample of 130
#people on three variables: body temperature (degrees
#Fahrenheit), gender (1: male, 2: female), and heart rate
#(beats per minute)
#Data analysis with Numpy
#Look inside the file BodyTemperature.txt (1=MALE,
#2=FEMALE)
#Read the file in numpy using the command
#np.genfromtxt() and put it into a numpy 2d array#
data=np.genfromtxt('BodyTemperature.txt',delimiter='\t',skip_header=1)
data

#Extract the number of Males and Females in the
#dataset
data_gender=data[:,1]
data_gender
data_gender= data_gender==1
np.count_nonzero(data_gender)
    

#Compute the overall mean for Temperature and
#HeartRate
data_TH=data[:,[0,2]]
data_TH
data_avg=np.mean(data_TH,axis=0)
data_avg
#Compute the mean, max and min of Temperature 
#and HeartRate for Male and Females separately
#and write the results on the file
#BodyTemperature_results.txt in a table format.

data[:,1]==1
M_mean=(np.mean(data[(data[:,1]==1)],axis=0))

M_max=np.max(data[(data[:,1]==1)],axis=0)
M_min=np.min(data[(data[:,1]==1)],axis=0)
W_mean=np.mean(data[(data[:,1]==2)],axis=0)
W_max=np.max(data[(data[:,1]==2)],axis=0)
W_min=np.min(data[(data[:,1]==2)],axis=0)
result=np.vstack([M_mean,M_max,M_min,W_mean,W_max,W_min])
result
np.savetxt('BodyTemperature_results.txt',result,delimiter=',')
#Normalize Temperature to 0-1 range
tf_max=np.max(data[:,0],axis=0)
tf_min=np.min(data[:,0],axis=0)
(data[:,0]-tf_min)/(tf_max-tf_min)
