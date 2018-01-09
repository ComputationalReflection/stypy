# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

r1 = 0 * np.nan
r2 = np.nan == np.nan
r3 = np.inf > np.nan
r4 = np.nan - np.nan
r5 = 0.3 == 3 * 0.1

Z = np.arange(11)

r6 = Z ** Z
r7 = 2 << Z >> 2
r8 = Z < - Z
r9 = 1j * Z
r10 = Z / 1 / 1
# Type error
#r11 = Z < Z > Z

r12 = np.array([0]) // np.array([0])

r13 = np.array([0]) // np.array([0.])
r14 = np.array([0]) / np.array([0])
r15 = np.array([0]) / np.array([0.])

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
