
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "assert keyword with an error in the condition"
4: 
5: if __name__ == '__main__':
6:     def f_error():
7:         # Type error
8:         return "a" / 3
9: 
10: 
11:     assert f_error()
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'assert keyword with an error in the condition')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def f_error(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f_error'
        module_type_store = module_type_store.open_function_context('f_error', 6, 4, False)
        
        # Passed parameters checking function
        f_error.stypy_localization = localization
        f_error.stypy_type_of_self = None
        f_error.stypy_type_store = module_type_store
        f_error.stypy_function_name = 'f_error'
        f_error.stypy_param_names_list = []
        f_error.stypy_varargs_param_name = None
        f_error.stypy_kwargs_param_name = None
        f_error.stypy_call_defaults = defaults
        f_error.stypy_call_varargs = varargs
        f_error.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'f_error', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f_error', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f_error(...)' code ##################

        str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 15), 'str', 'a')
        int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 21), 'int')
        # Applying the binary operator 'div' (line 8)
        result_div_4 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 15), 'div', str_2, int_3)
        
        # Assigning a type to the variable 'stypy_return_type' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type', result_div_4)
        
        # ################# End of 'f_error(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f_error' in the type store
        # Getting the type of 'stypy_return_type' (line 6)
        stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f_error'
        return stypy_return_type_5

    # Assigning a type to the variable 'f_error' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'f_error', f_error)
    # Evaluating assert statement condition
    
    # Call to f_error(...): (line 11)
    # Processing the call keyword arguments (line 11)
    kwargs_7 = {}
    # Getting the type of 'f_error' (line 11)
    f_error_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'f_error', False)
    # Calling f_error(args, kwargs) (line 11)
    f_error_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 11, 11), f_error_6, *[], **kwargs_7)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
