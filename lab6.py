# Variable and Operators
a = 47
b = 12.47
c = "Hello, Bryn"
d = True
e = False
f = [12, 13, 14]
g = 5.4e9
h = 1.45e-5

print(a)
print(type(a))
print(d+e)

a = 2; b = 9
c = a + b
print(c)

d = a < b
print(d)

e = 1 < 2 and 5 < 7
print(e)

a = "Hello" + "B"
print(a)
b = [11, 12, 14] + [0, 9, 7]
print(b)

# List and Tuple
a = [4, 6, 43, 3]
b = a[3]
print(b)
a.append(9)
print(a)
a[1] = 7
print(a)

a = (1, 5, 6)
b = a[1]
print(b)
a1, a2, a3 = a
print(a1, a2, a3)

# Dictionary
a = {"Pear":3, "Grape":4}
print(a["Pear"])
a["Grape"] = 7

a ["Melon"] = 15
print(a)

# if
a = 10

if a < 44:
    print("GOOD AM")
elif a < 47:
    print("Afternoon Partner")
elif a < 55:
    print("Evening Partner")
else:
    print("Night Captain")

# Loop
for i in[47, 10, 12]:
    print(i)
for i in range(4):
    print(i)
a = 1
while a < 3:
    print(a)
    a += 1

# Comprehension
a = [5, 3, 6, 7, 2, 1]
b = [c*3 for c in a]
print(b)
b = [c*3 for c in a if c < 7]
print(b)

# Function
def add1(a, b):
    c = a + b
    return c

def add2(a, b = 6):
    c = a + b
    return c

def add3(a, b, c):
    d = a + b + c
    return d

print(add1(4, 7))
print(add2(2))
t = (98,3, 12)
print(add3(*t))

# Class
class Parent:
    def __init__(self, a):
        self.a = a
    def add(self, b):
        print(self.a + b)
    def divide(self, b):
        print(self.a / b)

class Child(Parent):
    def subtract(self, b):
        print(self.a - b)
    def mult(self, b):
        print(self.a * b)

p = Parent(8)
p.add(6)
p.divide(2)

c = Child(3)
c.add(4)
c.divide(2)
c.subtract(1)
c.mult(3)
