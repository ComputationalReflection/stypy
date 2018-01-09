
# math: module
import math

# RightOps: RightOps

class RightOps:
    # self: instance
    # __float__() -> float 

    def __float__(self):
        # self: instance
        return 3.0

# r1: TypeError
r1 = math.pow(RightOps, 4)
# r2: TypeError
r2 = math.cos(RightOps)
# r3: float
r3 = math.pow(RightOps(), 4)
# r4: float
r4 = math.cos(RightOps())