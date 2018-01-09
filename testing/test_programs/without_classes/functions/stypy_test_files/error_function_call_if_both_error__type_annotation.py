
# function_1: function
# function_1(x: Compiler error in file 'error_function_call_if_both_error.py' (line 4, column 8):
        x /= 2
        ^
	Call to builtin_operators.div(str, int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_call_if_both_error.py' (line 10, column 5)
   Invocation to 'function_1(x: str)'
]Compiler error in file 'error_function_call_if_both_error.py' (line 6, column 8):
        x -= 2
        ^
	Call to builtin_operators.sub(str, int) is invalid.
	builtin_operators.sub(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ IterableObject \/ Instance defining __sub__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ RealNumber \/ <type 'complex'> \/ IterableObject \/ DynamicType \/ Instance defining __rsub__(parameter0)) expected.

Call stack: [
 - File '\error_function_call_if_both_error.py' (line 10, column 5)
   Invocation to 'function_1(x: str)'
]) -> int /\ function_1(x: Compiler error in file 'error_function_call_if_both_error.py' (line 4, column 8):
        x /= 2
        ^
	Call to builtin_operators.div(list[int], int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_call_if_both_error.py' (line 11, column 5)
   Invocation to 'function_1(x: list[int])'
]Compiler error in file 'error_function_call_if_both_error.py' (line 6, column 8):
        x -= 2
        ^
	Call to builtin_operators.sub(list[int], int) is invalid.
	builtin_operators.sub(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ IterableObject \/ Instance defining __sub__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ RealNumber \/ <type 'complex'> \/ IterableObject \/ DynamicType \/ Instance defining __rsub__(parameter0)) expected.

Call stack: [
 - File '\error_function_call_if_both_error.py' (line 11, column 5)
   Invocation to 'function_1(x: list[int])'
]) -> int 

def function_1(x):
    # a: int
    a = 0

    if (a > 0):
        # x: TypeError
        x /= 2
    else:
        # x: TypeError
        x -= 2

    return 3

# x: str; r1: int
r1 = function_1('a')
# x: list[int]; r2: int
r2 = function_1(range(5))
# function_2: function
# function_2(x: Compiler error in file 'error_function_call_if_both_error.py' (line 16, column 8):
        x /= 2
        ^
	Call to builtin_operators.div(str, int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_call_if_both_error.py' (line 22, column 5)
   Invocation to 'function_2(x: str)'
]Compiler error in file 'error_function_call_if_both_error.py' (line 19, column 8):
        x -= 2
        ^
	Call to builtin_operators.sub(str, int) is invalid.
	builtin_operators.sub(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ IterableObject \/ Instance defining __sub__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ RealNumber \/ <type 'complex'> \/ IterableObject \/ DynamicType \/ Instance defining __rsub__(parameter0)) expected.

Call stack: [
 - File '\error_function_call_if_both_error.py' (line 22, column 5)
   Invocation to 'function_2(x: str)'
]Compiler error in file 'error_function_call_if_both_error.py' (line 19, column 8):
        x -= 2
        ^
	Call to builtin_operators.sub(str, int) is invalid.
	builtin_operators.sub(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ IterableObject \/ Instance defining __sub__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ RealNumber \/ <type 'complex'> \/ IterableObject \/ DynamicType \/ Instance defining __rsub__(parameter0)) expected.

Call stack: [
 - File '\error_function_call_if_both_error.py' (line 22, column 5)
   Invocation to 'function_2(x: str)'
]) -> TypeError /\ function_2(x: Compiler error in file 'error_function_call_if_both_error.py' (line 16, column 8):
        x /= 2
        ^
	Call to builtin_operators.div(list[int], int) is invalid.
	builtin_operators.div(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ Instance defining __div__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ Integer \/ <type 'complex'> \/ <type 'float'> \/ RealNumber \/ DynamicType \/ Instance defining __rdiv__(parameter0)) expected.

Call stack: [
 - File '\error_function_call_if_both_error.py' (line 23, column 5)
   Invocation to 'function_2(x: list[int])'
]Compiler error in file 'error_function_call_if_both_error.py' (line 19, column 8):
        x -= 2
        ^
	Call to builtin_operators.sub(list[int], int) is invalid.
	builtin_operators.sub(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ IterableObject \/ Instance defining __sub__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ RealNumber \/ <type 'complex'> \/ IterableObject \/ DynamicType \/ Instance defining __rsub__(parameter0)) expected.

Call stack: [
 - File '\error_function_call_if_both_error.py' (line 23, column 5)
   Invocation to 'function_2(x: list[int])'
]Compiler error in file 'error_function_call_if_both_error.py' (line 19, column 8):
        x -= 2
        ^
	Call to builtin_operators.sub(list[int], int) is invalid.
	builtin_operators.sub(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ IterableObject \/ Instance defining __sub__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ RealNumber \/ <type 'complex'> \/ IterableObject \/ DynamicType \/ Instance defining __rsub__(parameter0)) expected.

Call stack: [
 - File '\error_function_call_if_both_error.py' (line 23, column 5)
   Invocation to 'function_2(x: list[int])'
]) -> TypeError 

def function_2(x):
    # a: int
    a = 0

    if (a > 0):
        # x: TypeError
        x /= 2
        return x
    else:
        # x: TypeError
        x -= 2
        return x


# x: str; r3: TypeError
r3 = function_2('a')
# r4: TypeError; x: list[int]
r4 = function_2(range(5))