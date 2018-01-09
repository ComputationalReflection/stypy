# http://www.scipy-lectures.org/intro/numpy/numpy.html
import numpy as np

a = np.triu(np.ones((3, 3)), 1)  # see help(np.triu)

r = a.T

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
