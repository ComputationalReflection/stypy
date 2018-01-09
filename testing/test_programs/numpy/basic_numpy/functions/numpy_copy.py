# http://www.python-course.eu/numpy.php

import numpy as np

x = np.array([[42, 22, 12], [44, 53, 66]], order='F')
y = x.copy()
x[0, 0] = 1001

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
