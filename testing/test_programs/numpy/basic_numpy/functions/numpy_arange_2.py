# http://www.python-course.eu/numpy.php

import time

import numpy as np

size_of_vec = 1000


def pure_python_version():
    t1 = time.time()
    X = range(size_of_vec)
    Y = range(size_of_vec)
    Z = []
    for i in range(len(X)):
        Z.append(X[i] + Y[i])

    l = locals().copy()
    for v in l:
        print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
    print("\n\n")
    return time.time() - t1


def numpy_version():
    t1 = time.time()
    X = np.arange(size_of_vec)
    Y = np.arange(size_of_vec)
    Z = X + Y

    l = locals().copy()
    for v in l:
        print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
    print("\n\n")
    return time.time() - t1


t1 = pure_python_version()
t2 = numpy_version()

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
