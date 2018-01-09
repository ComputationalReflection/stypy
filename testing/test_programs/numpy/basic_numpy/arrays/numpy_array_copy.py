# http://www.python-course.eu/numpy.php


import numpy as np

x = np.array([[42, 22, 12], [44, 53, 66]], order='F')
y = x.copy()
x[0, 0] = 1001

r = (x.flags['C_CONTIGUOUS'])
r2 = (y.flags['C_CONTIGUOUS'])

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
