import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

df = pd.read_csv("auto-mpg.data", sep='\s+', usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                 names=["mpg", "cyl", "dis", "hp", "weight", "acc", "model", "origin"])

data = []
for x in range(392):
    data.append([df.iloc[x, 1], df.iloc[x, 2], df.iloc[x, 3], df.iloc[x, 4], df.iloc[x, 5], df.iloc[x, 6], df.iloc[x, 7]])

scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)
print(data)
