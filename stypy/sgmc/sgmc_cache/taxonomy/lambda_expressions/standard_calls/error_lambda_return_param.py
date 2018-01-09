
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Returning a parameter from a function"
4: 
5: if __name__ == '__main__':
6:     import math
7: 
8:     f = lambda x: x
9: 
10:     y = f(3)
11:     print y * 2
12:     # Type error
13:     print y[12]
14:     # Type error
15:     print math.pow(f("a"), 3)
16: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Returning a parameter from a function')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 4))
    
    # 'import math' statement (line 6)
    import math

    import_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'math', math, module_type_store)
    
    
    # Assigning a Lambda to a Name (line 8):

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 8, 8, True)
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

        # Getting the type of 'x' (line 8)
        x_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 18), 'x')
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type', x_2)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 8)
        stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_3

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 8)
    _stypy_temp_lambda_1_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), '_stypy_temp_lambda_1')
    # Assigning a type to the variable 'f' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'f', _stypy_temp_lambda_1_4)
    
    # Assigning a Call to a Name (line 10):
    
    # Call to f(...): (line 10)
    # Processing the call arguments (line 10)
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'int')
    # Processing the call keyword arguments (line 10)
    kwargs_7 = {}
    # Getting the type of 'f' (line 10)
    f_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'f', False)
    # Calling f(args, kwargs) (line 10)
    f_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 10, 8), f_5, *[int_6], **kwargs_7)
    
    # Assigning a type to the variable 'y' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'y', f_call_result_8)
    # Getting the type of 'y' (line 11)
    y_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'y')
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 14), 'int')
    # Applying the binary operator '*' (line 11)
    result_mul_11 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 10), '*', y_9, int_10)
    
    
    # Obtaining the type of the subscript
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 12), 'int')
    # Getting the type of 'y' (line 13)
    y_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'y')
    # Obtaining the member '__getitem__' of a type (line 13)
    getitem___14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 10), y_13, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 13)
    subscript_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), getitem___14, int_12)
    
    
    # Call to pow(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Call to f(...): (line 15)
    # Processing the call arguments (line 15)
    str_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 21), 'str', 'a')
    # Processing the call keyword arguments (line 15)
    kwargs_20 = {}
    # Getting the type of 'f' (line 15)
    f_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 19), 'f', False)
    # Calling f(args, kwargs) (line 15)
    f_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 15, 19), f_18, *[str_19], **kwargs_20)
    
    int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 27), 'int')
    # Processing the call keyword arguments (line 15)
    kwargs_23 = {}
    # Getting the type of 'math' (line 15)
    math_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'math', False)
    # Obtaining the member 'pow' of a type (line 15)
    pow_17 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 10), math_16, 'pow')
    # Calling pow(args, kwargs) (line 15)
    pow_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), pow_17, *[f_call_result_21, int_22], **kwargs_23)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
