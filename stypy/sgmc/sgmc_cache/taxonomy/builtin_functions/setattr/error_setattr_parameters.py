
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "setattr method is present, but is invoked with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (AnyType, Str, AnyType) -> types.NoneType
7:     class Empty:
8:         pass
9: 
10: 
11:     # Call the builtin with incorrect number of parameters
12:     # Type error
13:     ret = setattr(Empty, "att", 0, None)
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'setattr method is present, but is invoked with a wrong number of parameters')
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
    
    # Assigning a Call to a Name (line 13):
    
    # Call to setattr(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'Empty' (line 13)
    Empty_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 18), 'Empty', False)
    str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 25), 'str', 'att')
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 32), 'int')
    # Getting the type of 'None' (line 13)
    None_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 35), 'None', False)
    # Processing the call keyword arguments (line 13)
    kwargs_7 = {}
    # Getting the type of 'setattr' (line 13)
    setattr_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'setattr', False)
    # Calling setattr(args, kwargs) (line 13)
    setattr_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), setattr_2, *[Empty_3, str_4, int_5, None_6], **kwargs_7)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', setattr_call_result_8)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
