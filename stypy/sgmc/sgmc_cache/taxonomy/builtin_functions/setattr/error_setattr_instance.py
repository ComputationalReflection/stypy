
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "setattr builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (AnyType, Str, AnyType) -> types.NoneType
7:     class Empty:
8:         pass
9: 
10: 
11:     # Type error
12:     ret = setattr(Empty, str, 0)
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'setattr builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Empty' class

    class Empty:
        pass

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 7, 4, False)
            # Assigning a type to the variable 'self' (line 8)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Empty.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Empty' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'Empty', Empty)
    
    # Assigning a Call to a Name (line 12):
    
    # Call to setattr(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'Empty' (line 12)
    Empty_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 18), 'Empty', False)
    # Getting the type of 'str' (line 12)
    str_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 25), 'str', False)
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 30), 'int')
    # Processing the call keyword arguments (line 12)
    kwargs_6 = {}
    # Getting the type of 'setattr' (line 12)
    setattr_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'setattr', False)
    # Calling setattr(args, kwargs) (line 12)
    setattr_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), setattr_2, *[Empty_3, str_4, int_5], **kwargs_6)
    
    # Assigning a type to the variable 'ret' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'ret', setattr_call_result_7)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
