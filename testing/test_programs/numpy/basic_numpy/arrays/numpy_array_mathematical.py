# http://cs231n.github.io/python-numpy-tutorial/

import numpy as np

x = np.array([[1, 2], [3, 4]])

r = np.sum(x)  # Compute sum of all elements; prints "10"
r2 = np.sum(x, axis=0)  # Compute sum of each column; prints "[4 6]"
r3 = np.sum(x, axis=1)  # Compute sum of each row; prints "[3 7]"

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
