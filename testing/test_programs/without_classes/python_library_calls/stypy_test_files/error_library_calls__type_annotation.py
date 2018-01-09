
# math: module
import math

# r1: TypeError
r1 = math.pow('a', 4)
# get_str: function
# get_str() -> str \/ int 

def get_str():

    if (r1 > 0):
        return 'hi'
    else:
        return 2


# r2: str \/ int
r2 = get_str()
# r3: float
r3 = math.pow(r2, 3)