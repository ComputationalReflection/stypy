import math

try:
    a = 3
except:
    a = "3"

r1 = a[6]  # No control about the possible values of a
r2 = math.fsum(a) # Not detected

