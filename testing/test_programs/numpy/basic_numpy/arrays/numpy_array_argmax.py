# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

Z = np.random.random(10)
Z[Z.argmax()] = 0

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
