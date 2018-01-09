# http://www.python-course.eu/numpy.php

import numpy as np

a = np.arange(1, 10)

# compare to range:
x = range(1, 10)

# some more arange examples:
x2 = np.arange(10.4)

x3 = np.arange(0.5, 10.4, 0.8)

x4 = np.arange(0.5, 10.4, 0.8, int)

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")