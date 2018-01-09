
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Checking the existence of __getslice__"
3: 
4: if __name__ == '__main__':
5:     class Dummy:
6:         pass
7: 
8: 
9:     d = Dummy()
10: 
11:     # Type error
12:     r = d[1:3]
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Checking the existence of __getslice__')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Dummy' class

    class Dummy:
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
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dummy.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Dummy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'Dummy', Dummy)
    
    # Assigning a Call to a Name (line 9):
    
    # Call to Dummy(...): (line 9)
    # Processing the call keyword arguments (line 9)
    kwargs_3 = {}
    # Getting the type of 'Dummy' (line 9)
    Dummy_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'Dummy', False)
    # Calling Dummy(args, kwargs) (line 9)
    Dummy_call_result_4 = invoke(stypy.reporting.localization.Localization(__file__, 9, 8), Dummy_2, *[], **kwargs_3)
    
    # Assigning a type to the variable 'd' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'd', Dummy_call_result_4)
    
    # Assigning a Subscript to a Name (line 12):
    
    # Obtaining the type of the subscript
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'int')
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 12), 'int')
    slice_7 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 12, 8), int_5, int_6, None)
    # Getting the type of 'd' (line 12)
    d_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'd')
    # Obtaining the member '__getitem__' of a type (line 12)
    getitem___9 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), d_8, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 12)
    subscript_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 12, 8), getitem___9, slice_7)
    
    # Assigning a type to the variable 'r' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'r', subscript_call_result_10)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
