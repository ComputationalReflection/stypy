# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

C = np.bincount([1, 1, 2, 3, 4, 4, 6])
A = np.repeat(np.arange(len(C)), C)
#
# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
