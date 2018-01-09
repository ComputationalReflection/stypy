
"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: class Foo:
4:     def __repr__(self):
5:         return "This is a test"
6: 
7: 
8: x = Foo()
9: 
10: y = repr(x)
11: 
12: z = repr(1+6+7)
13: print y
14: print z

"""

# Import the stypy library
from stypy import *

# Create the module type store
type_store = TypeStore(__file__)

################## Begin of the type inference program ##################

# Declaration of the 'Foo' class

class Foo:

    @norecursion
    def __repr__(type_of_self, localization, *varargs, **kwargs):
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        type_store.set_context('__repr__', 4, 4)
        # Type assignment (line 5)
        type_store.set_type_of(Localization(__file__, 5, 4), 'self', type_of_self)
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, type_store, 'Foo.__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            type_store.unset_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('__repr__', [], arguments)
        # Default return type storage variable (SSA)
        __stypy_ret_value = None
        
        # Begin of the function '__repr__' code
        __temp_728 = get_builtin_type(Localization(__file__, 5, 15), 'str', 'This is a test')
        __stypy_ret_value = union_type.UnionType.add(__stypy_ret_value, __temp_728)
        # End of the function '__repr__' code

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        # Destroy the context of function '__repr__'
        # Storing return type (line 4)
        type_store.store_return_type_of_current_context(__stypy_ret_value)
        type_store.unset_context()
        # Return type of the function
        return __stypy_ret_value


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        type_store.set_context('__init__', 3, 0)
        # Type assignment (line 4)
        type_store.set_type_of(Localization(__file__, 4, 0), 'self', type_of_self)
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, type_store, 'Foo.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            type_store.unset_context()
            return

        # Stacktrace push for error reporting
        localization.set_stack_trace('__init__', [], arguments)
        # Default return type storage variable (SSA)
        __stypy_ret_value = None
        
        # Begin of the function '__init__' code
        pass
        # End of the function '__init__' code

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        # Destroy the context of function '__init__'
        # Storing return type (line 3)
        type_store.store_return_type_of_current_context(__stypy_ret_value)
        type_store.unset_context()


# Type assignment (line 3)
type_store.set_type_of(Localization(__file__, 3, 0), 'Foo', Foo)

# Assignment to a Name from a Call

# Calling 'Foo' (line 8)
# Processing call keyword arguments (line 8)
__temp_730 = {}
# Getting the type of 'Foo' (line 8)
__temp_729 = type_store.get_type_of(Localization(__file__, 8, 4), 'Foo')
# Performing the call (line 8)
__temp_731 = __temp_729.invoke(Localization(__file__, 8, 4), *[], **__temp_730)

# Type assignment (line 8)
type_store.set_type_of(Localization(__file__, 8, 0), 'x', __temp_731)

# Assignment to a Name from a Call

# Calling 'repr' (line 10)
# Processing call arguments (line 10)
# Getting the type of 'x' (line 10)
__temp_733 = type_store.get_type_of(Localization(__file__, 10, 9), 'x')
# Processing call keyword arguments (line 10)
__temp_734 = {}
# Getting the type of 'repr' (line 10)
__temp_732 = type_store.get_type_of(Localization(__file__, 10, 4), 'repr')
# Performing the call (line 10)
__temp_735 = __temp_732.invoke(Localization(__file__, 10, 4), *[__temp_733], **__temp_734)

# Type assignment (line 10)
type_store.set_type_of(Localization(__file__, 10, 0), 'y', __temp_735)

# Assignment to a Name from a Call

# Calling 'repr' (line 12)
# Processing call arguments (line 12)


__temp_737 = get_builtin_type(Localization(__file__, 12, 9), 'int', 1)
__temp_738 = get_builtin_type(Localization(__file__, 12, 11), 'int', 6)
# Applying the '+' binary operator (line 12)
__temp_739 = operator(Localization(__file__, 12, 9), '+', __temp_737, __temp_738)

__temp_740 = get_builtin_type(Localization(__file__, 12, 13), 'int', 7)
# Applying the '+' binary operator (line 12)
__temp_741 = operator(Localization(__file__, 12, 12), '+', __temp_739, __temp_740)

# Processing call keyword arguments (line 12)
__temp_742 = {}
# Getting the type of 'repr' (line 12)
__temp_736 = type_store.get_type_of(Localization(__file__, 12, 4), 'repr')
# Performing the call (line 12)
__temp_743 = __temp_736.invoke(Localization(__file__, 12, 4), *[__temp_741], **__temp_742)

# Type assignment (line 12)
type_store.set_type_of(Localization(__file__, 12, 0), 'z', __temp_743)
# Getting the type of 'y' (line 13)
__temp_744 = type_store.get_type_of(Localization(__file__, 13, 6), 'y')
# Getting the type of 'z' (line 14)
__temp_745 = type_store.get_type_of(Localization(__file__, 14, 6), 'z')

################## End of the type inference program ##################

module_errors = TypeError.get_error_msgs()
module_warnings = TypeWarning.get_warning_msgs()
