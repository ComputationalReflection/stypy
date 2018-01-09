# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

Z = np.random.random((3, 3, 3))

Z2 = np.random.random((10, 10))
Zmin, Zmax = Z2.min(), Z2.max()

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
