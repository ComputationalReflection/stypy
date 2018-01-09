# https://docs.scipy.org/doc/numpy/reference/routines.math.html


import numpy as np

a = 2.1
a_min = 1
a_max = 3
v = 3.4
x = 3.5
xp = 5.6
fp = 5.8
x1 = 3.6
x2 = 3.7

# Miscellaneous
r1 = np.convolve(a, v)  # Returns the discrete, linear convolution of two one-dimensional sequences.
r2 = np.clip(a, a_min, a_max)  # Clip (limit) the values in an array.
r3 = np.sqrt(x)  # Return the positive square-root of an array, element-wise.
r4 = np.square(x)  # Return the element-wise square of the input.
r5 = np.absolute(x)  # Calculate the absolute value element-wise.
r6 = np.fabs(x)  # Compute the absolute values element-wise.
r7 = np.sign(x)  # Returns an element-wise indication of the sign of a number.
r8 = np.maximum(x1, x2)  # Element-wise maximum of array elements.
r9 = np.minimum(x1, x2)  # Element-wise minimum of array elements.
r10 = np.fmax(x1, x2)  # Element-wise maximum of array elements.
r11 = np.fmin(x1, x2)  # Element-wise minimum of array elements.
r12 = np.nan_to_num(x)  # Replace nan with zero and inf with finite numbers.
r13 = np.real_if_close(a)  # If complex input returns a real array if complex parts are close to zero.
# Type error
r14b = np.interp(x, xp, fp) 	#One-dimensional linear interpolation.

a = [2.1, 3.4, 5.6]
a_min = 1
a_max = 3
v = 3.4
x = [3.5, 3.6, 3.7, 3.8]
xp = 5.6
fp = 5.8
x1 = 3.6
x2 = 3.7

r14 = np.convolve(a, v)  # Returns the discrete, linear convolution of two one-dimensional sequences.
r15 = np.clip(a, a_min, a_max)  # Clip (limit) the values in an array.
r16 = np.sqrt(x)  # Return the positive square-root of an array, element-wise.
r17 = np.square(x)  # Return the element-wise square of the input.
r18 = np.absolute(x)  # Calculate the absolute value element-wise.
r19 = np.fabs(x)  # Compute the absolute values element-wise.
r20 = np.sign(x)  # Returns an element-wise indication of the sign of a number.
r21 = np.maximum(x1, x2)  # Element-wise maximum of array elements.
r22 = np.minimum(x1, x2)  # Element-wise minimum of array elements.
r23 = np.fmax(x1, x2)  # Element-wise maximum of array elements.
r24 = np.fmin(x1, x2)  # Element-wise minimum of array elements.
r25 = np.nan_to_num(x)  # Replace nan with zero and inf with finite numbers.
r26 = np.real_if_close(a)  # If complex input returns a real array if complex parts are close to zero.

xp = [1, 2, 3]
fp = [3, 2, 0]
r27 = np.interp(2.5, xp, fp)

r28 = np.interp([0, 1, 1.5, 2.72, 3.14], xp, fp)
r29 = np.array([3., 3., 2.5, 0.56, 0.])
UNDEF = -99.0
r30 = np.interp(3.14, xp, fp, right=UNDEF)

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
