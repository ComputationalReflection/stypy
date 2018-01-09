
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Get the type of a member of a function object"
3: 
4: if __name__ == '__main__':
5:     def f():
6:         pass
7: 
8: 
9:     r = getattr(f, 'func_name')
10:     print "|" + r + "|"
11: 
12:     # Type error
13:     print r / 3
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Get the type of a member of a function object')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def f(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 5, 4, False)
        
        # Passed parameters checking function
        f.stypy_localization = localization
        f.stypy_type_of_self = None
        f.stypy_type_store = module_type_store
        f.stypy_function_name = 'f'
        f.stypy_param_names_list = []
        f.stypy_varargs_param_name = None
        f.stypy_kwargs_param_name = None
        f.stypy_call_defaults = defaults
        f.stypy_call_varargs = varargs
        f.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'f', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f(...)' code ##################

        pass
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 5)
        stypy_return_type_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_2

    # Assigning a type to the variable 'f' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'f', f)
    
    # Assigning a Call to a Name (line 9):
    
    # Call to getattr(...): (line 9)
    # Processing the call arguments (line 9)
    # Getting the type of 'f' (line 9)
    f_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 16), 'f', False)
    str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 19), 'str', 'func_name')
    # Processing the call keyword arguments (line 9)
    kwargs_6 = {}
    # Getting the type of 'getattr' (line 9)
    getattr_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'getattr', False)
    # Calling getattr(args, kwargs) (line 9)
    getattr_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 9, 8), getattr_3, *[f_4, str_5], **kwargs_6)
    
    # Assigning a type to the variable 'r' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'r', getattr_call_result_7)
    str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'str', '|')
    # Getting the type of 'r' (line 10)
    r_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 16), 'r')
    # Applying the binary operator '+' (line 10)
    result_add_10 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 10), '+', str_8, r_9)
    
    str_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 20), 'str', '|')
    # Applying the binary operator '+' (line 10)
    result_add_12 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 18), '+', result_add_10, str_11)
    
    # Getting the type of 'r' (line 13)
    r_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'r')
    int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 14), 'int')
    # Applying the binary operator 'div' (line 13)
    result_div_15 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 10), 'div', r_13, int_14)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
