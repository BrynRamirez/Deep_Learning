from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
import matplotlib.pyplot as plt
import numpy as np

# data download
(trainX, trainY), (testX, testY) = mnist.load_data()

# convert to matrix format and normalize
trainX = trainX.reshape(trainX.shape[0], 28, 28, 1).astype('float32')/255
testX = testX.reshape(testX.shape[0], 28, 28, 1).astype('float32')/255

# one-hot encoding
trainY = np_utils.to_categorical(trainY, 10)
testY = np_utils.to_categorical(testY, 10)

# CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='sigmoid'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(64, kernel_size=(3, 3), activation='sigmoid'))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# train config
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(trainX, trainY, epochs=15, batch_size=500)

y_loss = history.history['accuracy']

plt.plot(np.arange(len(y_loss)), y_loss, marker='.', c='blue')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('train acc.')
plt.show()

print(model.evaluate(testX, testY)[1])
