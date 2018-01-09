# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

Z = np.random.randint(0, 10, (3, 3))

r = (Z[Z[:, 1].argsort()])

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
