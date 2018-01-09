
# function_1: function
# function_1(x: str) -> TypeError 

def function_1(x):
    return (x / 2)

# function_2: function
# function_2(x: list[int]) -> TypeError 

def function_2(x):
    return (x / 2)

# function_3: function
# function_3(x: int) -> int 

def function_3(x):
    return (x / 2)

# x: str; r1: TypeError
r1 = function_1('a')
# x: list[int]; r2: TypeError
r2 = function_2(range(5))
# x: int; r3: int
r3 = function_3(4)