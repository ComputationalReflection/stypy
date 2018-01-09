# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

Z = np.random.randint(0, 2, 100)
r1 = np.logical_not(Z, out=Z)

Z = np.random.uniform(-1.0, 1.0, 100)
r2 = np.negative(Z, out=Z)

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
