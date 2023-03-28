import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd

#print(print(tf.config.list_physical_devices('GPU')))
#tf.debugging.set_log_device_placement(True)

df = pd.read_csv("iris.data", usecols=[0, 1, 2, 3],
                     names=["1", "2", "3", "4"])

data = []
for x in range(150):
    data.append([df.iloc[x, 0], df.iloc[x, 1], df.iloc[x, 2], df.iloc[x, 3]])

with tf.device('/GPU:0'):
    averages = []

    print('Beginning Softmax Evaluation')

    for i in tf.range(100):
        i = tf.cast(i, tf.int64)
        trainX = tf.constant(data[15:151], dtype=tf.float32)
        trainY = tf.repeat([[1, 0, 0], [0, 1, 0], [0, 0, 1]], repeats=[45, 45, 45], axis=0)

        np.random.shuffle(data[0:15])
        testX = tf.constant(data[0:15], dtype=tf.float32)
        testY = tf.repeat([[1, 0, 0], [0, 1, 0], [0, 0, 1]], repeats=[5, 5, 5], axis=0)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(3, input_dim=4, activation='softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(trainX, trainY, epochs=600, batch_size=1, verbose=0)
        averages.append(model.evaluate(testX, testY)[1])
        print(f"Iteration: {i + 1}:100 complete")

    print(f'Total avg: {np.average(averages)}')
