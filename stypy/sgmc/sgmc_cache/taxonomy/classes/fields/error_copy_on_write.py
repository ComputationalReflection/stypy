
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Check the behavior of copy on write"
3: 
4: if __name__ == '__main__':
5:     class Foo:
6:         att = 3
7: 
8: 
9:     f = Foo()
10:     f2 = Foo()
11: 
12:     f.att = "str"
13: 
14:     # Type error
15:     print f.att + 3
16:     print f2.att + 3
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Check the behavior of copy on write')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Foo' class

    class Foo:
        pass

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 5, 4, False)
            # Assigning a type to the variable 'self' (line 6)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.__init__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            pass
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()

    
    # Assigning a type to the variable 'Foo' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'Foo', Foo)
    
    # Assigning a Num to a Name (line 6):
    int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 14), 'int')
    # Getting the type of 'Foo'
    Foo_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Foo')
    # Setting the type of the member 'att' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Foo_3, 'att', int_2)
    
    # Assigning a Call to a Name (line 9):
    
    # Call to Foo(...): (line 9)
    # Processing the call keyword arguments (line 9)
    kwargs_5 = {}
    # Getting the type of 'Foo' (line 9)
    Foo_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'Foo', False)
    # Calling Foo(args, kwargs) (line 9)
    Foo_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 9, 8), Foo_4, *[], **kwargs_5)
    
    # Assigning a type to the variable 'f' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'f', Foo_call_result_6)
    
    # Assigning a Call to a Name (line 10):
    
    # Call to Foo(...): (line 10)
    # Processing the call keyword arguments (line 10)
    kwargs_8 = {}
    # Getting the type of 'Foo' (line 10)
    Foo_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'Foo', False)
    # Calling Foo(args, kwargs) (line 10)
    Foo_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 10, 9), Foo_7, *[], **kwargs_8)
    
    # Assigning a type to the variable 'f2' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'f2', Foo_call_result_9)
    
    # Assigning a Str to a Attribute (line 12):
    str_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 12), 'str', 'str')
    # Getting the type of 'f' (line 12)
    f_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'f')
    # Setting the type of the member 'att' of a type (line 12)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), f_11, 'att', str_10)
    # Getting the type of 'f' (line 15)
    f_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'f')
    # Obtaining the member 'att' of a type (line 15)
    att_13 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 10), f_12, 'att')
    int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 18), 'int')
    # Applying the binary operator '+' (line 15)
    result_add_15 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 10), '+', att_13, int_14)
    
    # Getting the type of 'f2' (line 16)
    f2_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'f2')
    # Obtaining the member 'att' of a type (line 16)
    att_17 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 10), f2_16, 'att')
    int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 19), 'int')
    # Applying the binary operator '+' (line 16)
    result_add_19 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 10), '+', att_17, int_18)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
