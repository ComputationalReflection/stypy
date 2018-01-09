import math


class Empty:
    def __init__(self):
        pass


r1 = math.pow(Empty(), 3)  # Error: No __float__ method


class Ops:
    def __float__(self):
        return 3  # Don't return a float (even if the type promotes to float, a runtime error is reported


r2 = math.pow(Ops(), 3)  # Wrong __float__ (type conversion) method, not detected, runtime error


class WrongOps:
    def __float__(self):
        return "not a float"


r3 = math.pow(WrongOps(), 3)  # Runtime error, not reported


class EvenMoreWrongOps:
    def __float__(self, extra):
        return 3.0


r4 = math.pow(EvenMoreWrongOps(), 3)  # Not reported, even if the problem is parameter arity


class RightOps:
    def __float__(self):
        return 3.0


r5 = math.pow(RightOps(), 3)
