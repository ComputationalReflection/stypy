
# math: module
import math

# function: function
# function(x: int, **kwargs: dict{}) -> int /\ function(x: str, **kwargs: dict{}) -> str 

def function(x, **kwargs):
    return x

# y: int; x: int; kwargs: dict{}
y = function(3)
# r1: int
r1 = (y * 2)
# r2: TypeError
r2 = y[12]
# x: str; r3: TypeError; kwargs: dict{}
r3 = math.pow(function('a'), 3)