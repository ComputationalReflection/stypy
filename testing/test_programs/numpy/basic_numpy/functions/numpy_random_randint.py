# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

A = np.random.randint(0, 2, 5)
B = np.random.randint(0, 2, 5)
equal = np.allclose(A, B)

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
