# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

p, n = 10, 20
M = np.ones((p, n, n))
V = np.ones((p, n, 1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])

# It works, because:
# M is (p,n,n)
# V is (p,n,1)
# Thus, summing over the paired axes 0 and 0 (of M and V independently),
# and 2 and 1, to remain with a (n,1) vector.

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
