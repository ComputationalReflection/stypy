# http://www.scipy-lectures.org/intro/numpy/numpy.html

import numpy as np

x = np.array([1, 2, 3, -99, 5])

# One way to describe this is to create a masked array:

mx = np.ma.masked_array(x, mask=[0, 0, 0, 1, 0])

# Masked mean ignores masked data:

r = mx.mean()

mx[1] = 9

mx[1] = np.ma.masked

mx[1] = 9

# The mask is also available directly:

r2 = mx.mask

x2 = mx.filled(-1)

# The mask can also be cleared:

mx.mask = np.ma.nomask

r3 = mx

# Domain-aware functions

# The masked array package also contains domain-aware functions:

r4 = np.ma.log(np.array([1, 2, -1, -2, 3, -5]))

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
