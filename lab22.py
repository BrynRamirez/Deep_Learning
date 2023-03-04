from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout

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
model.add(Dropout(0.2))
model.add(MaxPool2D(pool_size=2))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='sigmoid'))
# model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(96, kernel_size=(3, 3), activation='sigmoid'))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# train config
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(trainX, trainY, epochs=30, batch_size=500)

model.save('my_model.h5')
del model
model = load_model('my_model.h5')

print(model.evaluate(testX, testY)[1])
