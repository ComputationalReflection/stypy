# https://docs.scipy.org/doc/numpy/reference/routines.math.html


import numpy as np

x = 2.1

# Other special functions
r1 = np.i0(x)  # Modified Bessel function of the first kind, order 0.
r2 = np.sinc(x)  # Return the sinc function.

x = [2.1, 2.2, 2.3]

r3 = np.i0(x)  # Modified Bessel function of the first kind, order 0.
r4 = np.sinc(x)  # Return the sinc function.

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
