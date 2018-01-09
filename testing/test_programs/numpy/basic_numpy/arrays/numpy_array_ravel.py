# http://www.scipy-lectures.org/intro/numpy/numpy.html
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
r = a.ravel()

r2 = a.T

r3 = a.T.ravel()
#
# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
