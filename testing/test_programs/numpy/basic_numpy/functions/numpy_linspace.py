# http://www.python-course.eu/numpy.php

import numpy as np

# 50 values between 1 and 10:
r = (np.linspace(1, 10))
# 7 values between 1 and 10:
r2 = (np.linspace(1, 10, 7))
# excluding the endpoint:
r3 = (np.linspace(1, 10, 7, endpoint=False))

samples, spacing = np.linspace(1, 10, retstep=True)
r4 = (spacing)
samples2, spacing = np.linspace(1, 10, 20, endpoint=True, retstep=True)
r5 = (spacing)
samples3, spacing = np.linspace(1, 10, 20, endpoint=False, retstep=True)
r6 = (spacing)

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
