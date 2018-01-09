
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Incorrect types passed on function calls"
4: 
5: if __name__ == '__main__':
6:     # Type error
7:     f = lambda x: x / 2
8: 
9:     # Type error
10:     f2 = lambda x: x / 2
11: 
12:     f3 = lambda x: x / 2
13: 
14:     y = f("a")
15:     y = f2(range(5))
16:     y = f3(4)
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Incorrect types passed on function calls')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Lambda to a Name (line 7):

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 7, 8, True)
        # Passed parameters checking function
        _stypy_temp_lambda_1.stypy_localization = localization
        _stypy_temp_lambda_1.stypy_type_of_self = None
        _stypy_temp_lambda_1.stypy_type_store = module_type_store
        _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
        _stypy_temp_lambda_1.stypy_param_names_list = ['x']
        _stypy_temp_lambda_1.stypy_varargs_param_name = None
        _stypy_temp_lambda_1.stypy_kwargs_param_name = None
        _stypy_temp_lambda_1.stypy_call_defaults = defaults
        _stypy_temp_lambda_1.stypy_call_varargs = varargs
        _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_1', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'x' (line 7)
        x_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 18), 'x')
        int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 22), 'int')
        # Applying the binary operator 'div' (line 7)
        result_div_4 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 18), 'div', x_2, int_3)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'stypy_return_type', result_div_4)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 7)
        stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_5

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 7)
    _stypy_temp_lambda_1_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), '_stypy_temp_lambda_1')
    # Assigning a type to the variable 'f' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'f', _stypy_temp_lambda_1_6)
    
    # Assigning a Lambda to a Name (line 10):

    @norecursion
    def _stypy_temp_lambda_2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_2'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_2', 10, 9, True)
        # Passed parameters checking function
        _stypy_temp_lambda_2.stypy_localization = localization
        _stypy_temp_lambda_2.stypy_type_of_self = None
        _stypy_temp_lambda_2.stypy_type_store = module_type_store
        _stypy_temp_lambda_2.stypy_function_name = '_stypy_temp_lambda_2'
        _stypy_temp_lambda_2.stypy_param_names_list = ['x']
        _stypy_temp_lambda_2.stypy_varargs_param_name = None
        _stypy_temp_lambda_2.stypy_kwargs_param_name = None
        _stypy_temp_lambda_2.stypy_call_defaults = defaults
        _stypy_temp_lambda_2.stypy_call_varargs = varargs
        _stypy_temp_lambda_2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_2', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_2', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'x' (line 10)
        x_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 19), 'x')
        int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 23), 'int')
        # Applying the binary operator 'div' (line 10)
        result_div_9 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 19), 'div', x_7, int_8)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'stypy_return_type', result_div_9)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_2' in the type store
        # Getting the type of 'stypy_return_type' (line 10)
        stypy_return_type_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_2'
        return stypy_return_type_10

    # Assigning a type to the variable '_stypy_temp_lambda_2' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), '_stypy_temp_lambda_2', _stypy_temp_lambda_2)
    # Getting the type of '_stypy_temp_lambda_2' (line 10)
    _stypy_temp_lambda_2_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), '_stypy_temp_lambda_2')
    # Assigning a type to the variable 'f2' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'f2', _stypy_temp_lambda_2_11)
    
    # Assigning a Lambda to a Name (line 12):

    @norecursion
    def _stypy_temp_lambda_3(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_3'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_3', 12, 9, True)
        # Passed parameters checking function
        _stypy_temp_lambda_3.stypy_localization = localization
        _stypy_temp_lambda_3.stypy_type_of_self = None
        _stypy_temp_lambda_3.stypy_type_store = module_type_store
        _stypy_temp_lambda_3.stypy_function_name = '_stypy_temp_lambda_3'
        _stypy_temp_lambda_3.stypy_param_names_list = ['x']
        _stypy_temp_lambda_3.stypy_varargs_param_name = None
        _stypy_temp_lambda_3.stypy_kwargs_param_name = None
        _stypy_temp_lambda_3.stypy_call_defaults = defaults
        _stypy_temp_lambda_3.stypy_call_varargs = varargs
        _stypy_temp_lambda_3.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_3', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_3', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'x' (line 12)
        x_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'x')
        int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 23), 'int')
        # Applying the binary operator 'div' (line 12)
        result_div_14 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 19), 'div', x_12, int_13)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'stypy_return_type', result_div_14)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_3' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_3'
        return stypy_return_type_15

    # Assigning a type to the variable '_stypy_temp_lambda_3' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), '_stypy_temp_lambda_3', _stypy_temp_lambda_3)
    # Getting the type of '_stypy_temp_lambda_3' (line 12)
    _stypy_temp_lambda_3_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), '_stypy_temp_lambda_3')
    # Assigning a type to the variable 'f3' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'f3', _stypy_temp_lambda_3_16)
    
    # Assigning a Call to a Name (line 14):
    
    # Call to f(...): (line 14)
    # Processing the call arguments (line 14)
    str_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 10), 'str', 'a')
    # Processing the call keyword arguments (line 14)
    kwargs_19 = {}
    # Getting the type of 'f' (line 14)
    f_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'f', False)
    # Calling f(args, kwargs) (line 14)
    f_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 14, 8), f_17, *[str_18], **kwargs_19)
    
    # Assigning a type to the variable 'y' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'y', f_call_result_20)
    
    # Assigning a Call to a Name (line 15):
    
    # Call to f2(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Call to range(...): (line 15)
    # Processing the call arguments (line 15)
    int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 17), 'int')
    # Processing the call keyword arguments (line 15)
    kwargs_24 = {}
    # Getting the type of 'range' (line 15)
    range_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 11), 'range', False)
    # Calling range(args, kwargs) (line 15)
    range_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 15, 11), range_22, *[int_23], **kwargs_24)
    
    # Processing the call keyword arguments (line 15)
    kwargs_26 = {}
    # Getting the type of 'f2' (line 15)
    f2_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'f2', False)
    # Calling f2(args, kwargs) (line 15)
    f2_call_result_27 = invoke(stypy.reporting.localization.Localization(__file__, 15, 8), f2_21, *[range_call_result_25], **kwargs_26)
    
    # Assigning a type to the variable 'y' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'y', f2_call_result_27)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to f3(...): (line 16)
    # Processing the call arguments (line 16)
    int_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'int')
    # Processing the call keyword arguments (line 16)
    kwargs_30 = {}
    # Getting the type of 'f3' (line 16)
    f3_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'f3', False)
    # Calling f3(args, kwargs) (line 16)
    f3_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), f3_28, *[int_29], **kwargs_30)
    
    # Assigning a type to the variable 'y' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'y', f3_call_result_31)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
