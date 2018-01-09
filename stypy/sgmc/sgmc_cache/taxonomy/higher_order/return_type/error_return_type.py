
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Infer the return type of the invocation of the function parameter"
3: 
4: if __name__ == '__main__':
5:     def higher_order(f, param1):
6:         return f(param1, param1)
7: 
8: 
9:     r = higher_order(lambda x, y: x + y, 3)
10: 
11:     # Type error
12:     print r + "str"
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Infer the return type of the invocation of the function parameter')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def higher_order(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'higher_order'
        module_type_store = module_type_store.open_function_context('higher_order', 5, 4, False)
        
        # Passed parameters checking function
        higher_order.stypy_localization = localization
        higher_order.stypy_type_of_self = None
        higher_order.stypy_type_store = module_type_store
        higher_order.stypy_function_name = 'higher_order'
        higher_order.stypy_param_names_list = ['f', 'param1']
        higher_order.stypy_varargs_param_name = None
        higher_order.stypy_kwargs_param_name = None
        higher_order.stypy_call_defaults = defaults
        higher_order.stypy_call_varargs = varargs
        higher_order.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'higher_order', ['f', 'param1'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'higher_order', localization, ['f', 'param1'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'higher_order(...)' code ##################

        
        # Call to f(...): (line 6)
        # Processing the call arguments (line 6)
        # Getting the type of 'param1' (line 6)
        param1_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 17), 'param1', False)
        # Getting the type of 'param1' (line 6)
        param1_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 25), 'param1', False)
        # Processing the call keyword arguments (line 6)
        kwargs_5 = {}
        # Getting the type of 'f' (line 6)
        f_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 15), 'f', False)
        # Calling f(args, kwargs) (line 6)
        f_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 6, 15), f_2, *[param1_3, param1_4], **kwargs_5)
        
        # Assigning a type to the variable 'stypy_return_type' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type', f_call_result_6)
        
        # ################# End of 'higher_order(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'higher_order' in the type store
        # Getting the type of 'stypy_return_type' (line 5)
        stypy_return_type_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'higher_order'
        return stypy_return_type_7

    # Assigning a type to the variable 'higher_order' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'higher_order', higher_order)
    
    # Assigning a Call to a Name (line 9):
    
    # Call to higher_order(...): (line 9)
    # Processing the call arguments (line 9)

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 9, 21, True)
        # Passed parameters checking function
        _stypy_temp_lambda_1.stypy_localization = localization
        _stypy_temp_lambda_1.stypy_type_of_self = None
        _stypy_temp_lambda_1.stypy_type_store = module_type_store
        _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
        _stypy_temp_lambda_1.stypy_param_names_list = ['x', 'y']
        _stypy_temp_lambda_1.stypy_varargs_param_name = None
        _stypy_temp_lambda_1.stypy_kwargs_param_name = None
        _stypy_temp_lambda_1.stypy_call_defaults = defaults
        _stypy_temp_lambda_1.stypy_call_varargs = varargs
        _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_1', ['x', 'y'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'x' (line 9)
        x_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 34), 'x', False)
        # Getting the type of 'y' (line 9)
        y_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 38), 'y', False)
        # Applying the binary operator '+' (line 9)
        result_add_11 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 34), '+', x_9, y_10)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 21), 'stypy_return_type', result_add_11)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 9)
        stypy_return_type_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 21), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_12

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 21), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 9)
    _stypy_temp_lambda_1_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 21), '_stypy_temp_lambda_1')
    int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 41), 'int')
    # Processing the call keyword arguments (line 9)
    kwargs_15 = {}
    # Getting the type of 'higher_order' (line 9)
    higher_order_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'higher_order', False)
    # Calling higher_order(args, kwargs) (line 9)
    higher_order_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 9, 8), higher_order_8, *[_stypy_temp_lambda_1_13, int_14], **kwargs_15)
    
    # Assigning a type to the variable 'r' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'r', higher_order_call_result_16)
    # Getting the type of 'r' (line 12)
    r_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'r')
    str_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'str', 'str')
    # Applying the binary operator '+' (line 12)
    result_add_19 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 10), '+', r_17, str_18)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
