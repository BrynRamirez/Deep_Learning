from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np


def hypothesis(X, w, b):
    return np.dot(X, w) + b


df = pd.read_csv("iris.data", usecols=[0, 1, 2, 3],
                 names=["1", "2", "3", "4"])

data = []
for x in range(100):
    data.append([df.iloc[x, 0], df.iloc[x, 1]])

averages = []

for x in range(100):
    np.random.shuffle(data)

    f10 = data[0:10]
    sample = data[10:100]

    trainX = sample
    trainy = np.array([
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1
    ])

    testX = f10
    testy = np.array([1, 1, 1, 1, 1,
                      -1, -1, -1, -1, -1])

    w = np.zeros(np.size(trainX, 1))
    b = 0
    alpha = 0.01

    for i in range(5000):
        w = w - alpha * (1 / len(trainX)) * np.dot(np.transpose(np.dot(trainX, w) + b - trainy), trainX)
        b = b - alpha * (1 / len(trainX)) * sum(np.dot(trainX, w) + b - trainy)

    print(w, b)
    avg = sum(np.sign(hypothesis(testX, w, b)) == testy) / len(testX)
    print(avg)
    averages.append(avg)

print("Total avg: ", np.average(averages))
