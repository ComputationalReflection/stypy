
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "issubclass builtin is invoked and its return type is used to call an non existing method"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Type, Type) -> <type 'bool'>
7: 
8:     class C:
9:         pass
10: 
11: 
12:     class D:
13:         pass
14: 
15: 
16:     # Call the builtin
17:     # No error
18:     ret = issubclass(C, D)
19: 
20:     # Type error
21:     ret.unexisting_method()
22: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'issubclass builtin is invoked and its return type is used to call an non existing method')
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
            module_type_store = module_type_store.open_function_context('__init__', 8, 4, False)
            # Assigning a type to the variable 'self' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'C' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'C', C)
    # Declaration of the 'D' class

    class D:
        pass

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 12, 4, False)
            # Assigning a type to the variable 'self' (line 13)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'D' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'D', D)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to issubclass(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'C' (line 18)
    C_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 21), 'C', False)
    # Getting the type of 'D' (line 18)
    D_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), 'D', False)
    # Processing the call keyword arguments (line 18)
    kwargs_5 = {}
    # Getting the type of 'issubclass' (line 18)
    issubclass_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 18)
    issubclass_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), issubclass_2, *[C_3, D_4], **kwargs_5)
    
    # Assigning a type to the variable 'ret' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'ret', issubclass_call_result_6)
    
    # Call to unexisting_method(...): (line 21)
    # Processing the call keyword arguments (line 21)
    kwargs_9 = {}
    # Getting the type of 'ret' (line 21)
    ret_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'ret', False)
    # Obtaining the member 'unexisting_method' of a type (line 21)
    unexisting_method_8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 4), ret_7, 'unexisting_method')
    # Calling unexisting_method(args, kwargs) (line 21)
    unexisting_method_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), unexisting_method_8, *[], **kwargs_9)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
