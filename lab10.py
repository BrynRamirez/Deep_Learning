import numpy as np
import matplotlib.pyplot as plt

data = np.array([[2.2, 6.14], [1.3, 4.72], [4.2, 11.17],
                 [5.8, 14.23], [3.4, 9.55], [8.7, 22.49]])

x = data[:, 0]
y = data[:, 1]

# initialization
w, b = 0, 0

# learning rate
alpha = 0.05

plt.scatter(x, y)
xl = np.linspace(0, 10, 100)

# Gradient Descent Algorithm
for i in range(2000):
    w = w - alpha * (1 / len(data)) * sum((w * x + b - y) * x)
    b = b - alpha * (1 / len(data)) * sum((w * x + b - y))

print("w = %f, b = %f" % (w, b))

plt.plot(xl, w * xl + b)
plt.show()
