# http://cs231n.github.io/python-numpy-tutorial/

import numpy as np

a = np.array([1, 2, 3])  # Create a rank 1 array
r = type(a)  # Prints "<type 'numpy.ndarray'>"
r2 = a.shape  # Prints "(3,)"
r3 = a[0], a[1], a[2]  # Prints "1 2 3"
a[0] = 5  # Change an element of the array
r4 = a  # Prints "[5, 2, 3]"

b = np.array([[1, 2, 3], [4, 5, 6]])  # Create a rank 2 array
r5 = b.shape  # Prints "(2, 3)"
r6 = b[0, 0], b[0, 1], b[1, 0]  # Prints "1 2 4"

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
