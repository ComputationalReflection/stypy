
# functionb: function
# functionb(x: Compiler error in file 'error_function_loops.py' (line 3, column 8):
        x /= 2
        ^
	Call to builtin_operators.div(str, int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_loops.py' (line 6, column 5)
   Invocation to 'functionb(x: str)'
] \/ str) -> Compiler error in file 'error_function_loops.py' (line 3, column 8):
        x /= 2
        ^
	Call to builtin_operators.div(str, int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_loops.py' (line 6, column 5)
   Invocation to 'functionb(x: str)'
] \/ str /\ functionb(x: Compiler error in file 'error_function_loops.py' (line 3, column 8):
        x /= 2
        ^
	Call to builtin_operators.div(list[int], int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_loops.py' (line 7, column 5)
   Invocation to 'functionb(x: list[int])'
] \/ list[int]) -> Compiler error in file 'error_function_loops.py' (line 3, column 8):
        x /= 2
        ^
	Call to builtin_operators.div(list[int], int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_loops.py' (line 7, column 5)
   Invocation to 'functionb(x: list[int])'
] \/ list[int] 

def functionb(x):
    # i: int
    for i in range(5):
        # x: TypeError
        x /= 2
    return x

# x: str; r1: Compiler error in file 'error_function_loops.py' (line 3, column 8):
        x /= 2
        ^
	Call to builtin_operators.div(str, int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_loops.py' (line 6, column 5)
   Invocation to 'functionb(x: str)'
]Compiler error in file 'error_function_loops.py' (line 3, column 8):
        x /= 2
        ^
	Call to builtin_operators.div(list[int], int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_loops.py' (line 7, column 5)
   Invocation to 'functionb(x: list[int])'
]Compiler error in file 'error_function_loops.py' (line 3, column 8):
        x /= 2
        ^
	Call to builtin_operators.div(list[int], int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_loops.py' (line 7, column 5)
   Invocation to 'functionb(x: list[int])'
]Compiler error in file 'error_function_loops.py' (line 3, column 8):
        x /= 2
        ^
	Call to builtin_operators.div(list[int], int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_loops.py' (line 7, column 5)
   Invocation to 'functionb(x: list[int])'
] \/ str
r1 = functionb('a')
# x: list[int]; r2: Compiler error in file 'error_function_loops.py' (line 3, column 8):
        x /= 2
        ^
	Call to builtin_operators.div(list[int], int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_loops.py' (line 7, column 5)
   Invocation to 'functionb(x: list[int])'
] \/ list[int]
r2 = functionb(range(5))