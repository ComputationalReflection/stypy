# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

Z = np.arange(9).reshape(3, 3)
for index, value in np.ndenumerate(Z):
    r = (index, value)
for index in np.ndindex(Z.shape):
    r2 = (index, Z[index])

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
