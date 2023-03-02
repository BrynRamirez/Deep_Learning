from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np

df = pd.read_csv("auto-mpg.data", sep='\s+', usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                 names=["mpg", "cyl", "dis", "hp", "weight", "acc", "model", "origin"])

data = []
for x in range(392):
    data.append([df.iloc[x, 1], df.iloc[x, 2], df.iloc[x, 3], df.iloc[x, 4], df.iloc[x, 5], df.iloc[x, 6],
                 df.iloc[x, 7], df.iloc[x, 0]])

scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)

np.random.shuffle(data)
splitArr = np.array_split(data, 10)  # splits into 10 arrays
# print(splitArr[0])
# print(splitArr[1:3])
combArr = np.concatenate((splitArr[1], splitArr[2], splitArr[3], splitArr[4], splitArr[5], splitArr[6], splitArr[7],
                          splitArr[8], splitArr[9]))
# print(combArr)
x = combArr[:, :-1]
# print(x)
y = combArr[:, -1]

# # initialization
w = np.array([0, 0, 0, 0, 0, 0, 0])
b = 0

# learning rate
alpha = 0.01

for i in range(50000):
    w = w - alpha * (1 / len(data)) * np.dot(np.transpose(np.dot(x, w) + b - y), x)
    b = b - alpha * (1 / len(data)) * sum(np.dot(x, w) + b - y)


# print(w, b)

# Find MSE
a_b = np.array(w, b)
print(a_b)
mse_data = splitArr[0]
mse_X = mse_data[:, :-1]
mse_Y = mse_data[:, -1]
print(mse_data)
# mse = sum(((splitArr[0] * x + splitArr[1]) - y) ** 2) / 352
# print(mse)
