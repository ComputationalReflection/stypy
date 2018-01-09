import math


def function(x, **kwargs):
    return x


y = function(3)
r1 = y * 2  # Correct and nothing reported
r2 = y[12]  # Unreported
r3 = math.pow(function("a"), 3)  # Unreported
