# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

Z = np.arange(10000)
x2 = np.random.shuffle(Z)
n = 5

# Slow
r = (Z[np.argsort(Z)[-n:]])

# Fast
r2 = (Z[np.argpartition(-Z, n)[:n]])

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
