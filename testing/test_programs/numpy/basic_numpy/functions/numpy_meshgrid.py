# http://www.labri.fr/perso/nrougier/teaching/numpy.100/

import numpy as np

Z = np.zeros((10, 10), [('x', float), ('y', float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0, 1, 10),
                             np.linspace(0, 1, 10))
#
# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
