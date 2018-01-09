# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

for dtype in [np.int8, np.int32, np.int64]:
    rf1 = (np.iinfo(dtype).min)
    rf2 = (np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
    rf3 = (np.finfo(dtype).min)
    rf4 = (np.finfo(dtype).max)
    rf5 = (np.finfo(dtype).eps)

r = np.finfo(np.float32).eps

r2 = np.finfo(np.float64).eps

r3 = np.float32(1e-8) + np.float32(1) == 1

r4 = np.float64(1e-8) + np.float64(1) == 1

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
