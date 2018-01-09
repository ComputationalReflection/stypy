import math


class RightOps:
    def __float__(self):
        return 3.0


r1 = math.pow(RightOps, 4)  # Not reported
r2 = math.cos(RightOps)  # Not reported

r3 = math.pow(RightOps(), 4)
r4 = math.cos(RightOps())
