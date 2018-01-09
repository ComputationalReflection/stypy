# https://docs.scipy.org/doc/numpy/reference/routines.math.html


import numpy as np

x = 2.1
x1 = 2.2
x2 = 2.3

# Floating point routines
r1 = np.signbit(x)  # Returns element-wise True where signbit is set (less than zero).
r2 = np.copysign(x1, x2)  # Change the sign of x1 to that of x2, element-wise.
r3 = np.frexp(x)  # Decompose the elements of x into mantissa and twos exponent.
#r4 = np.ldexp(x1, x2)  # Returns x1 * 2**x2, element-wise.

x = [2.1, 2.6]
x1 = [2.2, 2.7]
x2 = [2.3, 2.8]

r5 = np.signbit(x)  # Returns element-wise True where signbit is set (less than zero).
r6 = np.copysign(x1, x2)  # Change the sign of x1 to that of x2, element-wise.
r7 = np.frexp(x)  # Decompose the elements of x into mantissa and twos exponent.
#r8 = np.ldexp(x1, x2)  # Returns x1 * 2**x2, element-wise.

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
