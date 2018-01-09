
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Incorrect handling of a returned kwargs argument"
4: 
5: if __name__ == '__main__':
6:     f = lambda **kwargs: kwargs["val"]
7: 
8:     y2 = f(val="hi")
9:     # Type error
10:     y2 = y2.thisdonotexist()
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Incorrect handling of a returned kwargs argument')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Lambda to a Name (line 6):

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 6, 8, True)
        # Passed parameters checking function
        _stypy_temp_lambda_1.stypy_localization = localization
        _stypy_temp_lambda_1.stypy_type_of_self = None
        _stypy_temp_lambda_1.stypy_type_store = module_type_store
        _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
        _stypy_temp_lambda_1.stypy_param_names_list = []
        _stypy_temp_lambda_1.stypy_varargs_param_name = None
        _stypy_temp_lambda_1.stypy_kwargs_param_name = 'kwargs'
        _stypy_temp_lambda_1.stypy_call_defaults = defaults
        _stypy_temp_lambda_1.stypy_call_varargs = varargs
        _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', [], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_1', [], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Obtaining the type of the subscript
        str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 32), 'str', 'val')
        # Getting the type of 'kwargs' (line 6)
        kwargs_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 25), 'kwargs')
        # Obtaining the member '__getitem__' of a type (line 6)
        getitem___4 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 25), kwargs_3, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 6)
        subscript_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 6, 25), getitem___4, str_2)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type', subscript_call_result_5)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 6)
        stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_6

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 6)
    _stypy_temp_lambda_1_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), '_stypy_temp_lambda_1')
    # Assigning a type to the variable 'f' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'f', _stypy_temp_lambda_1_7)
    
    # Assigning a Call to a Name (line 8):
    
    # Call to f(...): (line 8)
    # Processing the call keyword arguments (line 8)
    str_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 15), 'str', 'hi')
    keyword_10 = str_9
    kwargs_11 = {'val': keyword_10}
    # Getting the type of 'f' (line 8)
    f_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 9), 'f', False)
    # Calling f(args, kwargs) (line 8)
    f_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 8, 9), f_8, *[], **kwargs_11)
    
    # Assigning a type to the variable 'y2' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'y2', f_call_result_12)
    
    # Assigning a Call to a Name (line 10):
    
    # Call to thisdonotexist(...): (line 10)
    # Processing the call keyword arguments (line 10)
    kwargs_15 = {}
    # Getting the type of 'y2' (line 10)
    y2_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'y2', False)
    # Obtaining the member 'thisdonotexist' of a type (line 10)
    thisdonotexist_14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 9), y2_13, 'thisdonotexist')
    # Calling thisdonotexist(args, kwargs) (line 10)
    thisdonotexist_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 10, 9), thisdonotexist_14, *[], **kwargs_15)
    
    # Assigning a type to the variable 'y2' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'y2', thisdonotexist_call_result_16)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
