# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np


def generate():
    for x in xrange(10):
        yield x


Z = np.fromiter(generate(), dtype=float, count=-1)

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
