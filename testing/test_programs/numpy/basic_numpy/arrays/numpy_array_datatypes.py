# http://cs231n.github.io/python-numpy-tutorial/

import numpy as np

x = np.array([1, 2])  # Let numpy choose the datatype
r = x.dtype  # Prints "int64"

x = np.array([1.0, 2.0])  # Let numpy choose the datatype
r2 = x.dtype  # Prints "float64"

x = np.array([1, 2], dtype=np.int64)  # Force a particular datatype
r3 = x.dtype  # Prints "int64"

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")