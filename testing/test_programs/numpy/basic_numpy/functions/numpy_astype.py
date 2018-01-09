# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

Z = np.arange(10, dtype=np.int32)
Z2 = Z.astype(np.float32, copy=False)

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")