# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

Z = np.random.random((10, 2))
X, Y = np.atleast_2d(Z[:, 0]), np.atleast_2d(Z[:, 1])
D = np.sqrt((X - X.T) ** 2 + (Y - Y.T) ** 2)


l = globals().copy()
for v in l:
    print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
