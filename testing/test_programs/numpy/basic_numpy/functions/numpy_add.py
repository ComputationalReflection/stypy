# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

Z = np.ones((16, 16))
k = 4

S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                    np.arange(0, Z.shape[1], k), axis=1)

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
