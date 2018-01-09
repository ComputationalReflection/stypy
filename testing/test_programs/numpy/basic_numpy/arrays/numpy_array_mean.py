# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

X = np.random.rand(5, 10)

# Recent versions of numpy
Y = X - X.mean(axis=1, keepdims=True)

# Older versions of numpy
Y2 = X - X.mean(axis=1).reshape(-1, 1)

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
