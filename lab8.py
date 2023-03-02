import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1.0, 1000)

# -log(x)
ln1 = np.log(x)
negln = -1 * ln1
fun1Y = negln

# -log(1-x)
ln2 = np.log(1 - x)
negln2 = -1 * ln2
fun2Y = negln2

plt.plot(x, fun1Y)
plt.plot(x, fun2Y)

plt.show()
