# http://telliott99.blogspot.com.es/2010/01/heres-question-on-so-about-how-to-make.html

import numpy as np

u = 15
b = np.zeros(u ** 2)
b.shape = (u, u)
w = b + 0x99

width = 20  # squares across of a single type
row1 = np.hstack([w, b] * width)
row2 = np.hstack([b, w] * width)
board = np.vstack([row1, row2] * width)
r = board.shape

# l = globals().copy()
# for v in l:
#     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
