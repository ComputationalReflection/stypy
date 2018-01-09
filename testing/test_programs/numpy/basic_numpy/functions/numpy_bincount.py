# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

Z = np.ones(10)
I = np.random.randint(0, len(Z), 20)
Z += np.bincount(I, minlength=len(Z))

X = [1, 2, 3, 4, 5, 6]
I = [1, 3, 9, 3, 4, 1]
F = np.bincount(I, X)

D = np.random.uniform(0, 1, 100)
S = np.random.randint(0, 10, 100)
D_sums = np.bincount(S, weights=D)
D_counts = np.bincount(S)
D_means = D_sums / D_counts

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
