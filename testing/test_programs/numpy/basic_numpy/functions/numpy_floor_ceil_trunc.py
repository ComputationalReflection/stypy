# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

Z = np.random.uniform(0, 10, 10)

r = (Z - Z % 1)
r2 = (np.floor(Z))
r3 = (np.ceil(Z) - 1)
r4 = (Z.astype(int))
r5 = (np.trunc(Z))

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
