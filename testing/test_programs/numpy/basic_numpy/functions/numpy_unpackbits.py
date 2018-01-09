# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B = ((I.reshape(-1, 1) & (2 ** np.arange(8))) != 0).astype(int)
r = (B[:, ::-1])

# Author: Daniel T. McDonald

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
r2 = (np.unpackbits(I[:, np.newaxis], axis=1))

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
