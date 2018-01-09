
# l: list[int \/ Compiler error in file 'error_augment_assignments.py' (line 2, column 7):
l[0] = l[0] - "a"  # Error detected
       ^
	Call to builtin_operators.sub(int, str) is invalid.
	builtin_operators.sub(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ IterableObject \/ Instance defining __sub__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ RealNumber \/ <type 'complex'> \/ IterableObject \/ DynamicType \/ Instance defining __rsub__(parameter0)) expected.

 \/ Compiler error in file 'error_augment_assignments.py' (line 3, column 0):
l[0] -= "a"  # Not detected
^
	Call to builtin_operators.sub(int, str) is invalid.
	builtin_operators.sub(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ IterableObject \/ Instance defining __sub__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ RealNumber \/ <type 'complex'> \/ IterableObject \/ DynamicType \/ Instance defining __rsub__(parameter0)) expected.

]
l = [1, 2, 4, 5]
# <container elements type>: int \/ Compiler error in file 'error_augment_assignments.py' (line 2, column 7):
l[0] = l[0] - "a"  # Error detected
       ^
	Call to builtin_operators.sub(int, str) is invalid.
	builtin_operators.sub(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ IterableObject \/ Instance defining __sub__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ RealNumber \/ <type 'complex'> \/ IterableObject \/ DynamicType \/ Instance defining __rsub__(parameter0)) expected.

 \/ Compiler error in file 'error_augment_assignments.py' (line 3, column 0):
l[0] -= "a"  # Not detected
^
	Call to builtin_operators.sub(int, str) is invalid.
	builtin_operators.sub(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ IterableObject \/ Instance defining __sub__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ RealNumber \/ <type 'complex'> \/ IterableObject \/ DynamicType \/ Instance defining __rsub__(parameter0)) expected.


l[0] = (l[0] - 'a')
# <container elements type>: int \/ Compiler error in file 'error_augment_assignments.py' (line 2, column 7):
l[0] = l[0] - "a"  # Error detected
       ^
	Call to builtin_operators.sub(int, str) is invalid.
	builtin_operators.sub(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ IterableObject \/ Instance defining __sub__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ RealNumber \/ <type 'complex'> \/ IterableObject \/ DynamicType \/ Instance defining __rsub__(parameter0)) expected.

 \/ Compiler error in file 'error_augment_assignments.py' (line 3, column 0):
l[0] -= "a"  # Not detected
^
	Call to builtin_operators.sub(int, str) is invalid.
	builtin_operators.sub(<type 'bool'> \/ <type 'complex'> \/ <type 'long'> \/ <type 'int'> \/ <type 'float'> \/ IterableObject \/ Instance defining __sub__(parameter0) \/ DynamicType, <type 'bool'> \/ Number \/ RealNumber \/ <type 'complex'> \/ IterableObject \/ DynamicType \/ Instance defining __rsub__(parameter0)) expected.


l[0] -= 'a'
# s: int
s = 3
# s: TypeError
s = (s + str(3))
# s: TypeError
s += str(3)
# s: TypeError
s += str(5)
# s: TypeError
s += str(7)