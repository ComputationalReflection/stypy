
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "issubclass builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Type, Type) -> <type 'bool'>
7: 
8: 
9:     class C:
10:         pass
11: 
12: 
13:     class D:
14:         pass
15: 
16: 
17:     # Call the builtin
18:     # No error
19:     ret = issubclass(C, D)
20: 
21:     # Call the builtin with incorrect types of parameters
22:     # Type error
23:     ret = issubclass(3, C)
24: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'issubclass builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'C' class

    class C:
        pass

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 9, 4, False)
            # Assigning a type to the variable 'self' (line 10)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'C.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'C' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'C', C)
    # Declaration of the 'D' class

    class D:
        pass

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 13, 4, False)
            # Assigning a type to the variable 'self' (line 14)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'D.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'D' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'D', D)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to issubclass(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'C' (line 19)
    C_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 21), 'C', False)
    # Getting the type of 'D' (line 19)
    D_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 24), 'D', False)
    # Processing the call keyword arguments (line 19)
    kwargs_5 = {}
    # Getting the type of 'issubclass' (line 19)
    issubclass_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 19)
    issubclass_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), issubclass_2, *[C_3, D_4], **kwargs_5)
    
    # Assigning a type to the variable 'ret' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'ret', issubclass_call_result_6)
    
    # Assigning a Call to a Name (line 23):
    
    # Call to issubclass(...): (line 23)
    # Processing the call arguments (line 23)
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'int')
    # Getting the type of 'C' (line 23)
    C_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'C', False)
    # Processing the call keyword arguments (line 23)
    kwargs_10 = {}
    # Getting the type of 'issubclass' (line 23)
    issubclass_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 23)
    issubclass_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 23, 10), issubclass_7, *[int_8, C_9], **kwargs_10)
    
    # Assigning a type to the variable 'ret' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'ret', issubclass_call_result_11)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
