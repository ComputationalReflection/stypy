# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
r = (np.linalg.det(C))

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
