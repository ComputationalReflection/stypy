
# functionb: function
# functionb(x: str \/ Compiler error in file 'error_function_call_if.py' (line 4, column 12):
        x = x / 2
            ^
	Call to builtin_operators.div(str, int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_call_if.py' (line 8, column 5)
   Invocation to 'functionb(x: str)'
]) -> str \/ Compiler error in file 'error_function_call_if.py' (line 4, column 12):
        x = x / 2
            ^
	Call to builtin_operators.div(str, int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_call_if.py' (line 8, column 5)
   Invocation to 'functionb(x: str)'
] /\ functionb(x: list[int] \/ Compiler error in file 'error_function_call_if.py' (line 4, column 12):
        x = x / 2
            ^
	Call to builtin_operators.div(list[int], int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_call_if.py' (line 9, column 5)
   Invocation to 'functionb(x: list[int])'
]) -> list[int] \/ Compiler error in file 'error_function_call_if.py' (line 4, column 12):
        x = x / 2
            ^
	Call to builtin_operators.div(list[int], int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_call_if.py' (line 9, column 5)
   Invocation to 'functionb(x: list[int])'
] 

def functionb(x):
    # a: int
    a = 0

    if (a > 0):
        # x: TypeError
        x = (x / 2)

    return x

# x: str; r1: str \/ Compiler error in file 'error_function_call_if.py' (line 4, column 12):
        x = x / 2
            ^
	Call to builtin_operators.div(str, int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_call_if.py' (line 8, column 5)
   Invocation to 'functionb(x: str)'
]Compiler error in file 'error_function_call_if.py' (line 4, column 12):
        x = x / 2
            ^
	Call to builtin_operators.div(list[int], int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_call_if.py' (line 9, column 5)
   Invocation to 'functionb(x: list[int])'
]Compiler error in file 'error_function_call_if.py' (line 4, column 12):
        x = x / 2
            ^
	Call to builtin_operators.div(list[int], int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_call_if.py' (line 9, column 5)
   Invocation to 'functionb(x: list[int])'
]Compiler error in file 'error_function_call_if.py' (line 4, column 12):
        x = x / 2
            ^
	Call to builtin_operators.div(list[int], int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_call_if.py' (line 9, column 5)
   Invocation to 'functionb(x: list[int])'
]
r1 = functionb('a')
# x: list[int]; r2: list[int] \/ Compiler error in file 'error_function_call_if.py' (line 4, column 12):
        x = x / 2
            ^
	Call to builtin_operators.div(list[int], int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_call_if.py' (line 9, column 5)
   Invocation to 'functionb(x: list[int])'
]
r2 = functionb(range(5))