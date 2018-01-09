
from math import fabs


try:
    from math import kos, sin, fabs
except Exception as ex:
    print ex

    from math import cos

r = cos(30)
r2 = sin(45)
print r2
