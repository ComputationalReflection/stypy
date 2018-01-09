# https://docs.scipy.org/doc/numpy/reference/routines.math.html

import numpy as np

x = 2.1
x1 = 3.4
x2 = 5.6

# Exponents and logarithms
r1 = np.exp(x)  # Calculate the exponential of all elements in the input array.
r2 = np.expm1(x)  # Calculate exp(x) - 1 for all elements in the array.
r3 = np.exp2(x)  # Calculate 2**p for all p in the input array.
r4 = np.log(x)  # Natural logarithm, element-wise.
r5 = np.log10(x)  # Return the base 10 logarithm of the input array, element-wise.
r6 = np.log2(x)  # Base-2 logarithm of x.
r7 = np.log1p(x)  # Return the natural logarithm of one plus the input array, element-wise.
r8 = np.logaddexp(x1, x2)  # Logarithm of the sum of exponentiations of the inputs.
r9 = np.logaddexp2(x1, x2)  # Logarithm of the sum of exponentiations of the inputs in base-2.

x = [2.1, 4.5, 6.7]
x1 = [3.4, 7.8, 9.8]
x2 = [5.6, 2.3, 6.7]

r10 = np.exp(x)  # Calculate the exponential of all elements in the input array.
r11 = np.expm1(x)  # Calculate exp(x) - 1 for all elements in the array.
r12 = np.exp2(x)  # Calculate 2**p for all p in the input array.
r13 = np.log(x)  # Natural logarithm, element-wise.
r14 = np.log10(x)  # Return the base 10 logarithm of the input array, element-wise.
r15 = np.log2(x)  # Base-2 logarithm of x.
r16 = np.log1p(x)  # Return the natural logarithm of one plus the input array, element-wise.
r17 = np.logaddexp(x1, x2)  # Logarithm of the sum of exponentiations of the inputs.
r18 = np.logaddexp2(x1, x2)  # Logarithm of the sum of exponentiations of the inputs in base-2.

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")