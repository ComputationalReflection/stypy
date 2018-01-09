# https://docs.scipy.org/doc/numpy/reference/routines.math.html


import numpy as np

x1 = 2.1
x2 = 3.4
x = 5.6

# Arithmetic operations
r1 = np.add(x1, x2)  # Add arguments element-wise.
r2 = np.reciprocal(x)  # Return the reciprocal of the argument, element-wise.
r3 = np.negative(x)  # Numerical negative, element-wise.
r4 = np.multiply(x1, x2)  # Multiply arguments element-wise.
r5 = np.divide(x1, x2)  # Divide arguments element-wise.
r6 = np.power(x1, x2)  # First array elements raised to powers from second array, element-wise.
r7 = np.subtract(x1, x2)  # Subtract arguments, element-wise.
r8 = np.true_divide(x1, x2)  # Returns a true division of the inputs, element-wise.
r9 = np.floor_divide(x1, x2)  # Return the largest integer smaller or equal to the division of the inputs.
r10 = np.fmod(x1, x2)  # Return the element-wise remainder of division.
r11 = np.mod(x1, x2)  # Return element-wise remainder of division.
r12 = np.modf(x)  # Return the fractional and integral parts of an array, element-wise.
r13 = np.remainder(x1, x2)  # Return element-wise remainder of division.

x1 = [2.1, 2.2, 2.3]
x2 = [3.4, 5.6, 7.8]
x = [5.6, 6.5, 7.8]

r14 = np.add(x1, x2)  # Add arguments element-wise.
r15 = np.reciprocal(x)  # Return the reciprocal of the argument, element-wise.
r16 = np.negative(x)  # Numerical negative, element-wise.
r17 = np.multiply(x1, x2)  # Multiply arguments element-wise.
r18 = np.divide(x1, x2)  # Divide arguments element-wise.
r19 = np.power(x1, x2)  # First array elements raised to powers from second array, element-wise.
r20 = np.subtract(x1, x2)  # Subtract arguments, element-wise.
r21 = np.true_divide(x1, x2)  # Returns a true division of the inputs, element-wise.
r22 = np.floor_divide(x1, x2)  # Return the largest integer smaller or equal to the division of the inputs.
r23 = np.fmod(x1, x2)  # Return the element-wise remainder of division.
r24 = np.mod(x1, x2)  # Return element-wise remainder of division.
r25 = np.modf(x)  # Return the fractional and integral parts of an array, element-wise.
r26 = np.remainder(x1, x2)  # Return element-wise remainder of division.

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")