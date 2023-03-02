import numpy as np

# Array
a = np.array([4, 5, 3, 5, 6])
print(a)

b = np.array([[5, 3, 1],
              [9, 5, 6]])
print(b)
c = np.array([[[3, 1],
               [4, 6],
               [9, 8]]])
print(c)
print()

# Shape and Size
print(np.shape(a))
print(np.size(a))
print(np.shape(b))
print(np.size(b))
print(np.shape(c))
print(np.size(c))
print()

# Matrices
print(np.zeros(5))
print(np.zeros((2, 3)))
print(np.ones(4))
print(np.random.rand(6))
print(np.arange(0, 1, 3))
print(np.linspace(0, 3, 4))
print()

# Reshape
a = np.array([5,3,4,6,2,9])
b = a.reshape(2,3)
print(b)
c = np.reshape(a, (3, -1))
print(c)
print()

# Addition
a = np.array([5,3,4,6,2,9])
b = np.array([7,3,6,6,9,3])
print(a+b)
print()

# Scalar Product
a = np.array([5,3,5])
b = 4
print(a*b)
print()

# Dot Product
a = np.array([5,3,4,7])
b = np.array([7,3,6,6])
print(np.dot(a,b))
print()
c = np.array([4,5,6,7,3,11]).reshape(3,2)
d = np.array([4,5,6,7,3,11]).reshape(2,3)
print(np.dot(c,d))
print()

# Access to elements
a = np.array([1,2,3,4,8,6])
print(a[3])
a[3] = 5
b = np.array([[4,5,6],[6,5,4]])
print(b[0,2])
print(b[0])
print()

# Slicing
a = np.array([1,2,4,8,6])
print(a[0:2])
b = np.array([[1,3,2],[3,1,2]])
print(b[:, 2])
print()

# Transpose
a = np.array([[1,3,2],[3,1,2]])
print(a)
print(a.T)
print()

# Numpy Functions
a = np.array([[1,3],[7,2]])
print(np.sum(a))
print(np.sum(a, axis=0))
print(np.sum(a, axis=1))
print(np.sum(a, axis=1, keepdims=True))
print(np.max(a))
print(np.argmax(a, axis=0))