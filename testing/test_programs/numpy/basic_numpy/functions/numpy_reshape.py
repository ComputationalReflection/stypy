# http://www.labri.fr/perso/nrougier/teaching/numpy.100/
import numpy as np

A = np.random.randint(0, 10, (3, 4, 3, 4))
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
