import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

df = pd.read_csv('iris.data', header=None)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

encoding = LabelEncoder()
encoding.fit(y)
y = encoding.transform(y)
y = tf.keras.utils.to_categorical(y)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(16, input_dim=4, activation='sigmoid'))
model.add(tf.keras.layers.Dense(14, activation='sigmoid'))
model.add(tf.keras.layers.Dense(7, activation='sigmoid'))
#model.add(tf.keras.layers.Dense(14, activation='sigmoid'))
model.add(tf.keras.layers.Dense(3, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=2000, batch_size=150)
