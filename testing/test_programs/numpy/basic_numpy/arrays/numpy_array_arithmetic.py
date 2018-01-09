# http://www.scipy-lectures.org/intro/numpy/numpy.html

import numpy as np

a = np.array([1, 2, 3, 4])
r = a + 1

r2 = 2 ** a

b = np.ones(4) + 1
r3 = a - b

r4 = a * b

j = np.arange(5)
r5 = 2 ** (j + 1) - j

c = np.ones((3, 3))
r6 = c * c  # NOT matrix multiplication!

# Matrix multiplication:

r = c.dot(c)

a = np.array([1, 2, 3, 4])
b = np.array([4, 2, 2, 4])
r7 = a == b

r8 = a > b

# Logical operations:

a = np.array([1, 1, 0, 0], dtype=bool)
b = np.array([1, 0, 1, 0], dtype=bool)
r9 = np.logical_or(a, b)

r10 = np.logical_and(a, b)

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
