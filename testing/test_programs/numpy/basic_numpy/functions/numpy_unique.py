# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

w, h = 16, 16
I = np.random.randint(0, 2, (h, w, 3)).astype(np.ubyte)
F = I[..., 0] * 256 * 256 + I[..., 1] * 256 + I[..., 2]
n = len(np.unique(F))
r = (np.unique(I))

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
