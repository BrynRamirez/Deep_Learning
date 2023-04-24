from urllib.request import urlretrieve
import gzip
import shutil
import os
import json
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt


# download and extract data
if not os.path.exists('CD_and_Vinyl.json.gz'):
    print('downloading data')
    url = 'https://jmcauley.ucsd.edu/data/amazon_v2/categoryFiles/Books.json.gz'
    dest = 'CD_and_Vinyl.json.gz'
    urlretrieve(url, dest)
    if not os.path.exists('CD_and_Vinyl.json'):
        with gzip.open('CD_and_Vinyl.json.gz', 'rb') as f_in:
            with open('CD_and_Vinyl.json', 'wb') as f_out:
                print('extracting data')
                shutil.copyfileobj(f_in, f_out)

text = []
overall = []
count = 0

with open('CD_and_Vinyl.json', 'r') as json_file:
    for line in json_file:
        data = json.loads(line.strip())
        if 'overall' in data and 'reviewText' in data:
            text.append(data['reviewText'])
            overall.append(int(data['overall']))
            count += 1
            if count > 1000:
                break


# create tokenizer
token = Tokenizer()
token.fit_on_texts(text)
#print(token.word_index)
text_len = max([len(i) for i in text])
word_size = len(token.word_index) + 1
print(word_size)

text = np.array(text)
overall = np.array(overall)
x_train, x_test, y_train, y_test = train_test_split(text, overall, test_size=0.20, random_state=104)
category = np.max(y_train)+1
cat = np.max(y_test)+1
print('y_train category', category)
print('y_test category', cat)
print(x_train[0])
print(y_train[0])
print(x_test[0])
print(y_test[0])

# add padding
x_train = token.texts_to_sequences(x_train)
x_train = pad_sequences(x_train, text_len)
print(x_train[0])
x_test = token.texts_to_sequences(x_test)
x_test = pad_sequences(x_test, text_len)
print(x_test[0])
y_train = to_categorical(y_train)
print(y_train[0])
y_test = to_categorical(y_test)
print(y_test[0])

# train LSTM model
model = Sequential()
model.add(Embedding(word_size, text_len))
model.add(LSTM(text_len, activation='tanh'))
model.add(Dense(6, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(x_train, y_train, epochs=40, validation_data=(x_test, y_test))
print('\n test accuracy: %.4f' % (model.evaluate(x_test, y_test)[1]))

# plot and save figure
plt.figure()
plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_acc'], label='value_accuracy')
plt.plot(history.history['val_loss'], label='value_loss')
plt.xlabel('Epoch')
plt.ylabel('Train Accuracy')
plt.legend()
plt.savefig('HW3_1_Plot.png')
plt.show()
