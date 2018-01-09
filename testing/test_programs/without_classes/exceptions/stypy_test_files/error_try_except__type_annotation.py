
# math: module
import math

try:
    # a: int
    a = 3
except:
    # a: str
    a = '3'
# r1: str
r1 = a[6]
# r2: TypeError
r2 = math.fsum(a)