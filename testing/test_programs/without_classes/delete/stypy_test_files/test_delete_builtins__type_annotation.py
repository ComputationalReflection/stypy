
# math: module
import math

# x_pre: float
x_pre = math.sin(4)
del math.sin
# x_post: TypeError
x_post = math.sin(4)