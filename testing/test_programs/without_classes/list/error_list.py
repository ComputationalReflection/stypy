import math

l1 = ["3", "4", 5, 6]

r1 = math.fsum(l1)  # Not detected
r2 = l1[2] / 3  # Not detected

