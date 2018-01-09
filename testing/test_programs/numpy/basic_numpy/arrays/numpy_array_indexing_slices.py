# http://cs231n.github.io/python-numpy-tutorial/

import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
row_r1 = a[1, :]  # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
r1, r2 = row_r1, row_r1.shape  # Prints "[5 6 7 8] (4,)"
r3, r4 = row_r2, row_r2.shape  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
r5, r6 = col_r1, col_r1.shape  # Prints "[ 2  6 10] (3,)"
r7, r8 = col_r2, col_r2.shape  # Prints "[[ 2]
#          [ 6]
#          [10]] (3, 1)"
#
# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
