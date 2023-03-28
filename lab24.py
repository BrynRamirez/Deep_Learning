import numpy as np
import tensorflow as tf
import keras
from keras.utils import np_utils
from keras.datasets import mnist
from keras import layers
from keras import Model
from functools import partial
import pathlib

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x, test_x = train_x/255.0, test_x/255.0
train_x = tf.expand_dims(train_x, 3)
test_x = tf.expand_dims(test_x, 3)
train_y = np_utils.to_categorical(train_y, 10)
test_y = np_utils.to_categorical(test_y, 10)

lenet_5_model = keras.models.Sequential([
    layers.Conv2D(6, kernel_size=5, activation='tanh', input_shape=train_x[0].shape, padding='same'),
    layers.AveragePooling2D(),
    layers.Conv2D(16, kernel_size=5, activation='tanh', padding='valid'),
    layers.AveragePooling2D(),
    layers.Flatten(),
    layers.Dense(120, activation='tanh'),
    layers.Dense(84, activation='tanh'),
    layers.Dense(10, activation='softmax')
])


alexnet_model = keras.models.Sequential([
    layers.Conv2D(96, 11, strides=4, activation='relu'),
    layers.MaxPool2D(3, 2),
    layers.Conv2D(256, 5, strides=1, padding='same', activation='relu'),
    layers.MaxPool2D(3, 2),
    layers.Conv2D(384, 3, strides=1, padding='same', activation='relu'),
    layers.Conv2D(256, 3, strides=1, padding='same', activation='relu'),
    layers.MaxPool2D(3, 2),
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

lenet_5_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
alexnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
lenet_5_model.fit(train_x, train_y, epochs=5)
lenet_5_model.evaluate(test_x, test_y)
