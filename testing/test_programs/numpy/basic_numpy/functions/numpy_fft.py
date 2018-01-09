# http://www.scipy-lectures.org/intro/numpy/numpy.html

import numpy as np

a = np.exp(2j * np.pi * np.arange(10))
fa = np.fft.fft(a)
r = np.set_printoptions(suppress=True)  # print small number as 0

a = np.exp(2j * np.pi * np.arange(3))
b = a[:, np.newaxis] + a[np.newaxis, :]
r2 = np.fft.fftn(b)

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
