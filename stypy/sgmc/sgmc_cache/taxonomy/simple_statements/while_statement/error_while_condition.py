
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "The condition is an error from a return type of a function"
3: 
4: if __name__ == '__main__':
5: 
6:     def err_func():
7:         # Type error
8:         return "a" / 3
9: 
10: 
11:     x = 0
12: 
13:     while err_func():
14:         x = 4
15: 
16:     x = x + 2
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'The condition is an error from a return type of a function')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def err_func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'err_func'
        module_type_store = module_type_store.open_function_context('err_func', 6, 4, False)
        
        # Passed parameters checking function
        err_func.stypy_localization = localization
        err_func.stypy_type_of_self = None
        err_func.stypy_type_store = module_type_store
        err_func.stypy_function_name = 'err_func'
        err_func.stypy_param_names_list = []
        err_func.stypy_varargs_param_name = None
        err_func.stypy_kwargs_param_name = None
        err_func.stypy_call_defaults = defaults
        err_func.stypy_call_varargs = varargs
        err_func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'err_func', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'err_func', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'err_func(...)' code ##################

        str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 15), 'str', 'a')
        int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 21), 'int')
        # Applying the binary operator 'div' (line 8)
        result_div_4 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 15), 'div', str_2, int_3)
        
        # Assigning a type to the variable 'stypy_return_type' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type', result_div_4)
        
        # ################# End of 'err_func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'err_func' in the type store
        # Getting the type of 'stypy_return_type' (line 6)
        stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'err_func'
        return stypy_return_type_5

    # Assigning a type to the variable 'err_func' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'err_func', err_func)
    
    # Assigning a Num to a Name (line 11):
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 8), 'int')
    # Assigning a type to the variable 'x' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'x', int_6)
    
    
    # Call to err_func(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_8 = {}
    # Getting the type of 'err_func' (line 13)
    err_func_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'err_func', False)
    # Calling err_func(args, kwargs) (line 13)
    err_func_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), err_func_7, *[], **kwargs_8)
    
    # Testing the type of an if condition (line 13)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 13, 4), err_func_call_result_9)
    # SSA begins for while statement (line 13)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Num to a Name (line 14):
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'int')
    # Assigning a type to the variable 'x' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'x', int_10)
    # SSA join for while statement (line 13)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 16):
    # Getting the type of 'x' (line 16)
    x_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'x')
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 12), 'int')
    # Applying the binary operator '+' (line 16)
    result_add_13 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 8), '+', x_11, int_12)
    
    # Assigning a type to the variable 'x' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'x', result_add_13)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
