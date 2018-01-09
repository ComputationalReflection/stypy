# https://docs.scipy.org/doc/numpy/reference/routines.math.html

import numpy as np

x = 2.1

# Hyperbolic functions
r1 = np.sinh(x)  # Hyperbolic sine, element-wise.
r2 = np.cosh(x)  # Hyperbolic cosine, element-wise.
r3 = np.tanh(x)  # Compute hyperbolic tangent element-wise.
r4 = np.arcsinh(x)  # Inverse hyperbolic sine element-wise.
r5 = np.arccosh(x)  # Inverse hyperbolic cosine, element-wise.
r6 = np.arctanh(x)  # Inverse hyperbolic tangent element-wise.

x = [0.1, 0.2, 0.3]

r7 = np.sinh(x)  # Hyperbolic sine, element-wise.
r8 = np.cosh(x)  # Hyperbolic cosine, element-wise.
r9 = np.tanh(x)  # Compute hyperbolic tangent element-wise.
r10 = np.arcsinh(x)  # Inverse hyperbolic sine element-wise.
r11 = np.arccosh(x)  # Inverse hyperbolic cosine, element-wise.
r12 = np.arctanh(x)  # Inverse hyperbolic tangent element-wise.

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")