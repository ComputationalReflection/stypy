# http://cs231n.github.io/python-numpy-tutorial/

import numpy as np

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

v = np.array([9, 10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
r = v.dot(w)
r2 = np.dot(v, w)

# Matrix / vector product; both produce the rank 1 array [29 67]
r3 = x.dot(v)
r4 = np.dot(x, v)

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
r5 = x.dot(y)
r6 = np.dot(x, y)

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
