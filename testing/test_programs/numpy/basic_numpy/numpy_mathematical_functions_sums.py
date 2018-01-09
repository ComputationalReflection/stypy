# https://docs.scipy.org/doc/numpy/reference/routines.math.html

import numpy as np

a = 2.1

# Sums, products, differences
r1 = np.prod(a)  # Return the product of array elements over a given axis.
r2 = np.sum(a)  # Sum of array elements over a given axis.
r3 = np.nansum(a)  # Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
r4 = np.cumprod(a)  # Return the cumulative product of elements along a given axis.
r5 = np.cumsum(a)  # Return the cumulative sum of the elements along a given axis.
# Type error
r6 = np.diff(a) 	#Calculate the n-th discrete difference along given axis.
r7 = np.ediff1d(a)  # The differences between consecutive elements of an array.
# Type error
r8 = np.gradient(a) 	#Return the gradient of an N-dimensional array.
# Type error
r9 = np.cross(a, a) 	#Return the cross product of two (arrays of) vectors.
# Type error
r10 = np.trapz(a) 	#Integrate along the given axis using the composite trapezoidal rule.

a = [1, 2, 3]
b = [8, 7, 5]

r11 = np.prod(a)  # Return the product of array elements over a given axis.
r12 = np.sum(a)  # Sum of array elements over a given axis.
r13 = np.nansum(a)  # Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
r14 = np.cumprod(a)  # Return the cumulative product of elements along a given axis.
r15 = np.cumsum(a)  # Return the cumulative sum of the elements along a given axis.
r16 = np.diff(a)  # Calculate the n-th discrete difference along given axis.
r17 = np.ediff1d(a)  # The differences between consecutive elements of an array.
r18 = np.gradient(a)  # Return the gradient of an N-dimensional array.
r19 = np.cross(a, b)  # Return the cross product of two (arrays of) vectors.
r20 = np.trapz(a)  # Integrate along the given axis using the composite trapezoidal rule.

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
