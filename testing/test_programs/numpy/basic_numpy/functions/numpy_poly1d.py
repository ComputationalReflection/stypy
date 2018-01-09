# http://www.scipy-lectures.org/intro/numpy/numpy.html

import numpy as np

p = np.poly1d([3, 2, -1])
r = p(0)

r2 = p.roots
r3 = p.order

x = np.linspace(0, 1, 20)
y = np.cos(x) + 0.3 * np.random.rand(20)
p = np.poly1d(np.polyfit(x, y, 3))

t = np.linspace(0, 1, 200)

p2 = np.polynomial.Polynomial([-1, 2, 3])  # coefs in different order!
r4 = p2(0)
r5 = p2.roots()
# Type error
r6 = p2.order
x2 = np.linspace(-1, 1, 2000)
y2 = np.cos(x) + 0.3 * np.random.rand(20)
p3 = np.polynomial.Chebyshev.fit(x, y, 90)

t2 = np.linspace(-1, 1, 200)

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
