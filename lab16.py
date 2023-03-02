from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np


def sigmoid(x):
    return 1 / (1 * np.e ** (-x))


df = pd.read_csv("iris.data", usecols=[0, 1, 2, 3],
                 names=["1", "2", "3", "4"])

data = []
for x in range(100):
    data.append([df.iloc[x, 0], df.iloc[x, 1]])

averages = []

for j in range(100):
    np.random.shuffle(data)

    f10 = data[0:10]
    sample = data[10:100]

    trainX = sample
    trainy = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1
    ])

    testX = f10
    testy = np.array([0, 1, 0, 1, 0,
                      1, 0, 1, 0, 1])

    w = np.zeros(np.size(trainX, 1))
    # print(w)
    lr = 0.05

    for i in range(5000):
        w_diff = np.dot(np.transpose(trainy - sigmoid(np.dot(trainX, w))), trainX)
        w = w * lr * w_diff

    avg = sum(np.round(sigmoid(np.dot(testX, w))) == testy) / np.size(testy)
    print(avg)
    averages.append(avg)

print("Total avg: ", np.average(averages))
