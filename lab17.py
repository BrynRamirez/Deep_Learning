import tensorflow as tf
import numpy as np
import pandas as pd

df = pd.read_csv("iris.data", usecols=[0, 1, 2, 3],
                 names=["1", "2", "3", "4"])
averages = []
data = []
for x in range(100):
    data.append([df.iloc[x, 0], df.iloc[x, 1], df.iloc[x, 2], df.iloc[x, 3]])

for j in range(100):
    np.random.shuffle(data)

    w = tf.Variable(tf.random.normal([4, 1]))
    b = tf.Variable(tf.random.normal([1, 1]))
    alpha = 0.05

    x = tf.constant(data[10:100], dtype=tf.float32)
    y = tf.constant([
        [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
        [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
        [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
        [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
        [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
        [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
        [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
        [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
        [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
        [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]],
        dtype=tf.float32)

    testx = tf.constant(data[0:10], dtype=tf.float32)
    testy = tf.constant([[0.], [1.], [0.], [1.], [0.],
                         [1.], [0.], [1.], [0.], [1.]],
                        dtype=tf.float32)


    def predict(x):
        # forward propagation
        out = tf.matmul(x, w)
        out = tf.add(out, b)
        out = tf.nn.sigmoid(out)
        return out


    def loss(y_predict, y):
        return tf.reduce_mean(tf.square(y_predict - y))


    for i in range(10000):
        with tf.GradientTape() as t:
            current_loss = loss(predict(x), y)
            dW, db = t.gradient(current_loss, [w, b])
            w.assign_sub(alpha * dW)
            b.assign_sub(alpha * db)

    #print(w.numpy(), b.numpy())
    avg = tf.round(predict(testx)).numpy()
    averages.append(avg)
    #print(j, " of 99")

print("total average: ", np.average(averages))
