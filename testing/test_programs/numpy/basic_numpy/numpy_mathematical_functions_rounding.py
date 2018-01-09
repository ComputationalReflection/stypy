# https://docs.scipy.org/doc/numpy/reference/routines.math.html


import numpy as np

x = 2.1
a = 3.45646

# Rounding
r1 = np.around(a)  # Evenly round to the given number of decimals.
r2 = np.round_(a)  # Round an array to the given number of decimals.
r3 = np.rint(x)  # Round elements of the array to the nearest integer.
r4 = np.fix(x)  # Round to nearest integer towards zero.
r5 = np.floor(x)  # Return the floor of the input, element-wise.
r6 = np.ceil(x)  # Return the ceiling of the input, element-wise.
r7 = np.trunc(x)  # Return the truncated value of the input, element-wise.

x = [2.1, 3.4]
a = [3.45646, 5.6432564]

r8 = np.around(a)  # Evenly round to the given number of decimals.
r9 = np.round_(a)  # Round an array to the given number of decimals.
r10 = np.rint(x)  # Round elements of the array to the nearest integer.
r11 = np.fix(x)  # Round to nearest integer towards zero.
r12 = np.floor(x)  # Return the floor of the input, element-wise.
r13 = np.ceil(x)  # Return the ceiling of the input, element-wise.
r14 = np.trunc(x)  # Return the truncated value of the input, element-wise.


# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
