
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Check each type of the function parameter"
3: 
4: if __name__ == '__main__':
5:     def func(p1, p2):
6:         # Type error
7:         return p1 / p2
8: 
9: 
10:     def higher_order(f, param1, param2):
11:         return f(param1, param2)
12: 
13: 
14:     higher_order(func, "3", "4")
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Check each type of the function parameter')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 5, 4, False)
        
        # Passed parameters checking function
        func.stypy_localization = localization
        func.stypy_type_of_self = None
        func.stypy_type_store = module_type_store
        func.stypy_function_name = 'func'
        func.stypy_param_names_list = ['p1', 'p2']
        func.stypy_varargs_param_name = None
        func.stypy_kwargs_param_name = None
        func.stypy_call_defaults = defaults
        func.stypy_call_varargs = varargs
        func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func', ['p1', 'p2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'func', localization, ['p1', 'p2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'func(...)' code ##################

        # Getting the type of 'p1' (line 7)
        p1_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 15), 'p1')
        # Getting the type of 'p2' (line 7)
        p2_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 20), 'p2')
        # Applying the binary operator 'div' (line 7)
        result_div_4 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 15), 'div', p1_2, p2_3)
        
        # Assigning a type to the variable 'stypy_return_type' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'stypy_return_type', result_div_4)
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 5)
        stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_5

    # Assigning a type to the variable 'func' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'func', func)

    @norecursion
    def higher_order(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'higher_order'
        module_type_store = module_type_store.open_function_context('higher_order', 10, 4, False)
        
        # Passed parameters checking function
        higher_order.stypy_localization = localization
        higher_order.stypy_type_of_self = None
        higher_order.stypy_type_store = module_type_store
        higher_order.stypy_function_name = 'higher_order'
        higher_order.stypy_param_names_list = ['f', 'param1', 'param2']
        higher_order.stypy_varargs_param_name = None
        higher_order.stypy_kwargs_param_name = None
        higher_order.stypy_call_defaults = defaults
        higher_order.stypy_call_varargs = varargs
        higher_order.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'higher_order', ['f', 'param1', 'param2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'higher_order', localization, ['f', 'param1', 'param2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'higher_order(...)' code ##################

        
        # Call to f(...): (line 11)
        # Processing the call arguments (line 11)
        # Getting the type of 'param1' (line 11)
        param1_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 17), 'param1', False)
        # Getting the type of 'param2' (line 11)
        param2_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 25), 'param2', False)
        # Processing the call keyword arguments (line 11)
        kwargs_9 = {}
        # Getting the type of 'f' (line 11)
        f_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 15), 'f', False)
        # Calling f(args, kwargs) (line 11)
        f_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 11, 15), f_6, *[param1_7, param2_8], **kwargs_9)
        
        # Assigning a type to the variable 'stypy_return_type' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type', f_call_result_10)
        
        # ################# End of 'higher_order(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'higher_order' in the type store
        # Getting the type of 'stypy_return_type' (line 10)
        stypy_return_type_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'higher_order'
        return stypy_return_type_11

    # Assigning a type to the variable 'higher_order' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'higher_order', higher_order)
    
    # Call to higher_order(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'func' (line 14)
    func_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 17), 'func', False)
    str_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 23), 'str', '3')
    str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 28), 'str', '4')
    # Processing the call keyword arguments (line 14)
    kwargs_16 = {}
    # Getting the type of 'higher_order' (line 14)
    higher_order_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'higher_order', False)
    # Calling higher_order(args, kwargs) (line 14)
    higher_order_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), higher_order_12, *[func_13, str_14, str_15], **kwargs_16)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
