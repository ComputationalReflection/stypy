# http: //www.python-course.eu/numpy.php

import numpy as np

r = np.identity(4)

r2 = np.identity(4, dtype=int)  # equivalent to np.identity(3, int)

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
