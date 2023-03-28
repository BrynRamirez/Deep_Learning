from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras import optimizers
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

opt1 = optimizers.Adam(learning_rate=0.001)
opt2 = optimizers.SGD(learning_rate=0.1)
opt3 = optimizers.Adagrad(learning_rate=0.01)
opt4 = optimizers.RMSprop(learning_rate=0.001)
opt5 = optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)

# CNN and configurations
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='sigmoid'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(64, kernel_size=(3, 3), activation='sigmoid'))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=opt1, metrics=['accuracy'])
history = model.fit(trainX, trainY, epochs=15, batch_size=500)
y_loss1 = history.history['accuracy']
plt.plot(np.arange(len(y_loss1)), y_loss1, marker='.', c='blue')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='sigmoid'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(64, kernel_size=(3, 3), activation='sigmoid'))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=opt2, metrics=['accuracy'])
history = model.fit(trainX, trainY, epochs=15, batch_size=500)
y_loss2 = history.history['accuracy']
plt.plot(np.arange(len(y_loss2)), y_loss2, marker='.', c='red')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='sigmoid'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(64, kernel_size=(3, 3), activation='sigmoid'))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=opt3, metrics=['accuracy'])
history = model.fit(trainX, trainY, epochs=15, batch_size=500)
y_loss3 = history.history['accuracy']
plt.plot(np.arange(len(y_loss3)), y_loss3, marker='.', c='orange')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='sigmoid'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(64, kernel_size=(3, 3), activation='sigmoid'))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=opt4, metrics=['accuracy'])
history = model.fit(trainX, trainY, epochs=15, batch_size=500)
y_loss4 = history.history['accuracy']
plt.plot(np.arange(len(y_loss4)), y_loss4, marker='.', c='black')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='sigmoid'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(64, kernel_size=(3, 3), activation='sigmoid'))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=opt5, metrics=['accuracy'])
history = model.fit(trainX, trainY, epochs=15, batch_size=500)
y_loss5 = history.history['accuracy']
plt.plot(np.arange(len(y_loss5)), y_loss5, marker='.', c='yellow')

plt.grid()
plt.xlabel('epoch')
plt.ylabel('train acc.')
plt.legend(['Adam', 'SGD', 'Adagrad', 'RMSProp', 'Nesterov Momentum'])
plt.show()

#print(model.evaluate(testX, testY)[1])
