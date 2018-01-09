# http://www.scipy-lectures.org/intro/numpy/numpy.html

import numpy as np

x, y = np.ogrid[0:5, 0:5]

r1, r2 = x.shape, y.shape

distance = np.sqrt(x ** 2 + y ** 2)

x, y = np.mgrid[0:4, 0:4]
#
# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
