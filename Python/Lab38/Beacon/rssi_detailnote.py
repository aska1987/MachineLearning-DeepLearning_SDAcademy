# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 03:03:45 2018

@author: sun
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib

sns.set()    
matplotlib.rcParams['figure.figsize'] = [12, 8]

path = './rssi.csv'
data = pd.read_csv(path)

"""
RSSI신호 세기는 0에 가까울 수록(적을수록) 더 세다.
차트의 크기를 8,20으로 지정
차트를 그리기위한 subplot을 4행 2열로 지정
"""

fig = plt.figure(figsize=(8, 20))
from itertools import product

axs = fig.subplots(4,2)
for pair, ax in zip(product((1,2), ("A","B","C","D")), axs.flatten()):
    (floor, ap) = pair
    mask = (data.z == floor) & (data.ap == ap)
    signal = data[mask][["signal", "x", "y"]]
    ax.scatter(signal.x, signal.y, c=signal.signal)
    ax.set_title("Floor: %s AP: %s" %(floor, ap))
    
plt.savefig('rssi1.png')
plt.show()
    
# 각 샘플링 위치와 AP의 유클리드 거리를 구한다.
ap_coordinates = {"A": (23, 17, 2), "B": (23, 41, 2), "C" : (1, 15, 2), "D": (1, 41, 2)}
g = data.groupby(["x", "y", "z", "ap"])
def dist(df):
    ap_coords = ap_coordinates[df.iloc[0].ap]
    x, y, z = ap_coords
    df["distance"] = np.sqrt((df.x - x) ** 2 + (df.y - y) ** 2 + (df.z - z) ** 2)
    return df
print(data.head(5))
data = g.apply(dist)
print(data.head(5))

"""
수치가 적을 수록 신호가 강한 것임.
구해지는 수치에 산란, 반산, 방해, 간섭 등의 오차가 있을 수 있음
RSSI신호 세기는 0에 가까울 수록(적을수록) 더 세다.
차트의 크기를 18,60으로 지정
"""

fig, axes = plt.subplots(4,2, figsize=(18, 16))
for pair, ax in zip(product((1,2), ("A","B","C","D")), axes.flatten()):
    (floor, ap) = pair
    mask = (data.z == floor) & (data.ap == ap)
    signal = data[mask][["signal", "distance"]]
    ax.plot(signal.distance, signal.signal, '.')
    ax.set_ylabel("RSSI")
    ax.set_title("Floor %s, AP: %s" %(floor, ap))
    
plt.savefig('rssi2.png')
plt.show()
    
print(data.head(5))
"""
최소, 최대, 평균, 중앙값을 기준으로 data값을 교차해 그림
"""

fig, axes = plt.subplots(2,2, figsize=(18, 16))
estimators = [np.min, np.max, np.mean, np.median]
for ax, estimator in zip(axes.flatten(), estimators):
    mask = (data.z == 2) & (data.ap == "A")
    signal = data[mask][["signal", "distance"]]
    sns.regplot("distance", "signal", data=data, 
                x_estimator=estimator, x_bins=100, ax=ax)
    ax.set_title(estimator.__name__)

plt.savefig('rssi3.png')
plt.show()
