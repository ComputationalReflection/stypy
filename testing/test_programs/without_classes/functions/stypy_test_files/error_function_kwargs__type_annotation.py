
# function: function
# function(x: int, **kwargs: dict[{str: str}]) -> int \/ Compiler error in file 'error_function_kwargs.py' (line 6, column 15):
        return kwargs[0]  # Should warn about None
               ^
	No value is associated to key type 'int'.

Call stack: [
 - File '\error_function_kwargs.py' (line 9, column 4)
   Invocation to 'function(x: int, **kwargs={val: str})'
] 

def function(x, **kwargs):
    # a: int
    a = 0

    if (a > 0):
        return int(x)
    else:
        return kwargs[0]


# y: int \/ Compiler error in file 'error_function_kwargs.py' (line 6, column 15):
        return kwargs[0]  # Should warn about None
               ^
	No value is associated to key type 'int'.

Call stack: [
 - File '\error_function_kwargs.py' (line 9, column 4)
   Invocation to 'function(x: int, **kwargs={val: str})'
]; x: int; kwargs: dict[{str: str}]
y = function(3, val='hi')
# y2: TypeError
y2 = y.thisdonotexist()