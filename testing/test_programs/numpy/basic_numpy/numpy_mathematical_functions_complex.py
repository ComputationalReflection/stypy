# https://docs.scipy.org/doc/numpy/reference/routines.math.html

import numpy as np

z = 2.1j + 3

# Handling complex numbers
r1 = np.angle([z])  # Return the angle of the complex argument.
r2 = np.real(z)  # Return the real part of the elements of the array.
r3 = np.imag(z)  # Return the imaginary part of the elements of the array.
r4 = np.conj(z)  # Return the complex conjugate, element-wise.

z = [2.1j + 3, 3.1j + 3, 4.1j + 3]

r5 = np.angle(z)  # Return the angle of the complex argument.
r6 = np.real(z)  # Return the real part of the elements of the array.
r7 = np.imag(z)  # Return the imaginary part of the elements of the array.
r8 = np.conj(z)  # Return the complex conjugate, element-wise.


# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")