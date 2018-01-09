# http://www.python-course.eu/numpy.php

import numpy as np

F = np.array([1, 1, 2, 3, 5, 8, 13, 21])
V = np.array([3.4, 6.9, 99.8, 12.8])

r = F.dtype
r2 = V.dtype
r3 = np.ndim(F)
r4 = np.ndim(V)

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
