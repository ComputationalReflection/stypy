
# cos: builtin_function_or_method
from math import cos as aliased

# aliased: list[]
aliased = []
# alias: function
# alias() -> None 

def alias():
    # r: TypeError
    r = aliased(0.5)

alias()