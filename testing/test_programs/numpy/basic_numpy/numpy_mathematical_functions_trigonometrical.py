# https://docs.scipy.org/doc/numpy/reference/routines.math.html

import numpy as np

x = 2.1

# Trigonometric functions
r1 = np.sin(x)  # Trigonometric sine, element-wise.
r2 = np.cos(x)  # Cosine element-wise.
r3 = np.tan(x)  # Compute tangent element-wise.
r4 = np.arcsin(x/10)  # Inverse sine, element-wise.
r5 = np.arccos(x/10)  # Trigonometric inverse cosine, element-wise.
r6 = np.arctan(x)  # Trigonometric inverse tangent, element-wise.

r7 = np.hypot(3 * np.ones((3, 3)), 4 * np.ones((3, 3)))  # Given the 'legs' of a right triangle, return its hypotenuse.

o1 = np.array([-1, +1, +1, -1])
o2 = np.array([-1, -1, +1, +1])
r8 = np.arctan2(o2, o1) * 180 / np.pi  # Element-wise arc tangent of x1/x2 choosing the quadrant correctly.

r9 = np.degrees(x)  # Convert angles from radians to degrees.
r10 = np.radians(x)  # Convert angles from degrees to radians.

phase = np.linspace(0, np.pi, num=5)
phase[3:] += np.pi
r11 = np.unwrap(phase)  # Unwrap by changing deltas between values to 2*pi complement.

r12 = np.deg2rad(x)  # Convert angles from degrees to radians.
r13 = np.rad2deg(x)  # Convert angles from radians to degrees.

x = [1, 2, 3, 4]
x10 = [0.1, 0.2, 0.3, 0.4]

r14 = np.sin(x)  # Trigonometric sine, element-wise.
r15 = np.cos(x)  # Cosine element-wise.
r16 = np.tan(x)  # Compute tangent element-wise.
r17 = np.arcsin(x10)  # Inverse sine, element-wise.
r18 = np.arccos(x10)  # Trigonometric inverse cosine, element-wise.
r19 = np.arctan(x)  # Trigonometric inverse tangent, element-wise.

r20 = np.degrees(x)  # Convert angles from radians to degrees.
r21 = np.radians(x)  # Convert angles from degrees to radians.

r22 = np.deg2rad(x)  # Convert angles from degrees to radians.
r23 = np.rad2deg(x)  # Convert angles from radians to degrees.

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
