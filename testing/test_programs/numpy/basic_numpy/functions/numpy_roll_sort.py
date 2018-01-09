# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

faces = np.random.randint(0, 100, (10, 3))
F = np.roll(faces.repeat(2, axis=1), -1, axis=1)
F = F.reshape(len(F) * 3, 2)
F = np.sort(F, axis=1)
G = F.view(dtype=[('p0', F.dtype), ('p1', F.dtype)])
G = np.unique(G)
#
# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
