# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np
from numpy.lib import stride_tricks


def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    l = locals().copy()
    for v in l:
        print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
    print "\n\n"
    return stride_tricks.as_strided(a, shape=shape, strides=strides)


Z = rolling(np.arange(10), 3)
#
# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
