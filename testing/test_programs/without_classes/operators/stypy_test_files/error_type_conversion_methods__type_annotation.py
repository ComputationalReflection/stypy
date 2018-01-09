
# math: module
import math

# Empty: Empty

class Empty:
    # __init__() -> None 

    def __init__(self):
        # self: instance
        pass

# r1: TypeError
r1 = math.pow(Empty(), 3)
# Ops: Ops

class Ops:
    # self: instance
    # __float__() -> int 

    def __float__(self):
        # self: instance
        return 3

# r2: TypeError
r2 = math.pow(Ops(), 3)
# WrongOps: WrongOps

class WrongOps:
    # self: instance
    # __float__() -> str 

    def __float__(self):
        # self: instance
        return 'not a float'

# r3: TypeError
r3 = math.pow(WrongOps(), 3)
# EvenMoreWrongOps: EvenMoreWrongOps

class EvenMoreWrongOps:
    # self: instance
    # __float__(extra: Compiler error in file 'error_type_conversion_methods.py' (line 33, column 5):
r4 = math.pow(EvenMoreWrongOps(), 3)  # Not reported, even if the problem is parameter arity
     ^
	Insufficient number of arguments for EvenMoreWrongOps.__float__: Cannot find a value for argument number 1 ('extra');.

) -> None 

    def __float__(self, extra):
        # self: instance
        return 3.0

# r4: TypeError; extra: TypeError
r4 = math.pow(EvenMoreWrongOps(), 3)
# RightOps: RightOps

class RightOps:
    # self: instance
    # __float__() -> float 

    def __float__(self):
        # self: instance
        return 3.0

# r5: float
r5 = math.pow(RightOps(), 3)