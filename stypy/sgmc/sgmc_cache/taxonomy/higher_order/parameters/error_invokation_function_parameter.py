
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Check the invocation over the function parameter"
3: 
4: if __name__ == '__main__':
5:     def higher_order(f, param1):
6:         ret = f(param1, param1)
7:         return ret + 4
8: 
9: 
10:     def higher_order_wrong(f, param1):
11:         # Type error
12:         ret = param1(f)
13:         return ret
14: 
15: 
16:     # Type error
17:     higher_order(lambda x, y: x + str(y), 3)
18: 
19:     higher_order_wrong(lambda x, y: x + str(y), 3)
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Check the invocation over the function parameter')
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

        
        # Assigning a Call to a Name (line 6):
        
        # Call to f(...): (line 6)
        # Processing the call arguments (line 6)
        # Getting the type of 'param1' (line 6)
        param1_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 16), 'param1', False)
        # Getting the type of 'param1' (line 6)
        param1_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 24), 'param1', False)
        # Processing the call keyword arguments (line 6)
        kwargs_5 = {}
        # Getting the type of 'f' (line 6)
        f_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 14), 'f', False)
        # Calling f(args, kwargs) (line 6)
        f_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 6, 14), f_2, *[param1_3, param1_4], **kwargs_5)
        
        # Assigning a type to the variable 'ret' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'ret', f_call_result_6)
        # Getting the type of 'ret' (line 7)
        ret_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 15), 'ret')
        int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 21), 'int')
        # Applying the binary operator '+' (line 7)
        result_add_9 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 15), '+', ret_7, int_8)
        
        # Assigning a type to the variable 'stypy_return_type' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'stypy_return_type', result_add_9)
        
        # ################# End of 'higher_order(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'higher_order' in the type store
        # Getting the type of 'stypy_return_type' (line 5)
        stypy_return_type_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'higher_order'
        return stypy_return_type_10

    # Assigning a type to the variable 'higher_order' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'higher_order', higher_order)

    @norecursion
    def higher_order_wrong(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'higher_order_wrong'
        module_type_store = module_type_store.open_function_context('higher_order_wrong', 10, 4, False)
        
        # Passed parameters checking function
        higher_order_wrong.stypy_localization = localization
        higher_order_wrong.stypy_type_of_self = None
        higher_order_wrong.stypy_type_store = module_type_store
        higher_order_wrong.stypy_function_name = 'higher_order_wrong'
        higher_order_wrong.stypy_param_names_list = ['f', 'param1']
        higher_order_wrong.stypy_varargs_param_name = None
        higher_order_wrong.stypy_kwargs_param_name = None
        higher_order_wrong.stypy_call_defaults = defaults
        higher_order_wrong.stypy_call_varargs = varargs
        higher_order_wrong.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'higher_order_wrong', ['f', 'param1'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'higher_order_wrong', localization, ['f', 'param1'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'higher_order_wrong(...)' code ##################

        
        # Assigning a Call to a Name (line 12):
        
        # Call to param1(...): (line 12)
        # Processing the call arguments (line 12)
        # Getting the type of 'f' (line 12)
        f_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 21), 'f', False)
        # Processing the call keyword arguments (line 12)
        kwargs_13 = {}
        # Getting the type of 'param1' (line 12)
        param1_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 14), 'param1', False)
        # Calling param1(args, kwargs) (line 12)
        param1_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 12, 14), param1_11, *[f_12], **kwargs_13)
        
        # Assigning a type to the variable 'ret' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'ret', param1_call_result_14)
        # Getting the type of 'ret' (line 13)
        ret_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', ret_15)
        
        # ################# End of 'higher_order_wrong(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'higher_order_wrong' in the type store
        # Getting the type of 'stypy_return_type' (line 10)
        stypy_return_type_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'higher_order_wrong'
        return stypy_return_type_16

    # Assigning a type to the variable 'higher_order_wrong' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'higher_order_wrong', higher_order_wrong)
    
    # Call to higher_order(...): (line 17)
    # Processing the call arguments (line 17)

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 17, 17, True)
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

        # Getting the type of 'x' (line 17)
        x_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 30), 'x', False)
        
        # Call to str(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'y' (line 17)
        y_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 38), 'y', False)
        # Processing the call keyword arguments (line 17)
        kwargs_21 = {}
        # Getting the type of 'str' (line 17)
        str_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 34), 'str', False)
        # Calling str(args, kwargs) (line 17)
        str_call_result_22 = invoke(stypy.reporting.localization.Localization(__file__, 17, 34), str_19, *[y_20], **kwargs_21)
        
        # Applying the binary operator '+' (line 17)
        result_add_23 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 30), '+', x_18, str_call_result_22)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 17), 'stypy_return_type', result_add_23)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 17), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_24

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 17), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 17)
    _stypy_temp_lambda_1_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 17), '_stypy_temp_lambda_1')
    int_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 42), 'int')
    # Processing the call keyword arguments (line 17)
    kwargs_27 = {}
    # Getting the type of 'higher_order' (line 17)
    higher_order_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'higher_order', False)
    # Calling higher_order(args, kwargs) (line 17)
    higher_order_call_result_28 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), higher_order_17, *[_stypy_temp_lambda_1_25, int_26], **kwargs_27)
    
    
    # Call to higher_order_wrong(...): (line 19)
    # Processing the call arguments (line 19)

    @norecursion
    def _stypy_temp_lambda_2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_2'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_2', 19, 23, True)
        # Passed parameters checking function
        _stypy_temp_lambda_2.stypy_localization = localization
        _stypy_temp_lambda_2.stypy_type_of_self = None
        _stypy_temp_lambda_2.stypy_type_store = module_type_store
        _stypy_temp_lambda_2.stypy_function_name = '_stypy_temp_lambda_2'
        _stypy_temp_lambda_2.stypy_param_names_list = ['x', 'y']
        _stypy_temp_lambda_2.stypy_varargs_param_name = None
        _stypy_temp_lambda_2.stypy_kwargs_param_name = None
        _stypy_temp_lambda_2.stypy_call_defaults = defaults
        _stypy_temp_lambda_2.stypy_call_varargs = varargs
        _stypy_temp_lambda_2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_2', ['x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_2', ['x', 'y'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'x' (line 19)
        x_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 36), 'x', False)
        
        # Call to str(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'y' (line 19)
        y_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 44), 'y', False)
        # Processing the call keyword arguments (line 19)
        kwargs_33 = {}
        # Getting the type of 'str' (line 19)
        str_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 40), 'str', False)
        # Calling str(args, kwargs) (line 19)
        str_call_result_34 = invoke(stypy.reporting.localization.Localization(__file__, 19, 40), str_31, *[y_32], **kwargs_33)
        
        # Applying the binary operator '+' (line 19)
        result_add_35 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 36), '+', x_30, str_call_result_34)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'stypy_return_type', result_add_35)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_2' in the type store
        # Getting the type of 'stypy_return_type' (line 19)
        stypy_return_type_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_2'
        return stypy_return_type_36

    # Assigning a type to the variable '_stypy_temp_lambda_2' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), '_stypy_temp_lambda_2', _stypy_temp_lambda_2)
    # Getting the type of '_stypy_temp_lambda_2' (line 19)
    _stypy_temp_lambda_2_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), '_stypy_temp_lambda_2')
    int_38 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 48), 'int')
    # Processing the call keyword arguments (line 19)
    kwargs_39 = {}
    # Getting the type of 'higher_order_wrong' (line 19)
    higher_order_wrong_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'higher_order_wrong', False)
    # Calling higher_order_wrong(args, kwargs) (line 19)
    higher_order_wrong_call_result_40 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), higher_order_wrong_29, *[_stypy_temp_lambda_2_37, int_38], **kwargs_39)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
