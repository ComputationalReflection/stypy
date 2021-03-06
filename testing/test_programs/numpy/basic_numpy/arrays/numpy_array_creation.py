# http://cs231n.github.io/python-numpy-tutorial/

import numpy as np

a = np.zeros((2, 2))  # Create an array of all zeros

b = np.ones((1, 2))  # Create an array of all ones

c = np.full((2, 2), 7)  # Create a constant array

d = np.eye(2)  # Create a 2x2 identity matrix

e = np.random.random((2, 2))  # Create an array filled with random values

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
