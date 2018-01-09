
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "sorted builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (IterableObject) -> <type 'list'>
7:     # (IterableObject, Has__call__) -> <type 'list'>
8:     # (IterableObject, Has__call__, Has__call__) -> <type 'list'>
9:     # (IterableObject, Has__call__, Has__call__, <type bool>) -> <type 'list'>
10:     # (Str) -> <type 'list'>
11:     # (Str, Has__call__) -> <type 'list'>
12:     # (Str, Has__call__, Has__call__) -> <type 'list'>
13:     # (Str, Has__call__, Has__call__, <type bool>) -> <type 'list'>
14:     import types
15: 
16:     # Type error
17:     ret = sorted(list, lambda x, y: str(x) == str(y))
18:     # Type error
19:     ret = sorted("str", types.FunctionType)
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'sorted builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 4))
    
    # 'import types' statement (line 14)
    import types

    import_module(stypy.reporting.localization.Localization(__file__, 14, 4), 'types', types, module_type_store)
    
    
    # Assigning a Call to a Name (line 17):
    
    # Call to sorted(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'list' (line 17)
    list_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 17), 'list', False)

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 17, 23, True)
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

        
        
        # Call to str(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'x' (line 17)
        x_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 40), 'x', False)
        # Processing the call keyword arguments (line 17)
        kwargs_6 = {}
        # Getting the type of 'str' (line 17)
        str_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 36), 'str', False)
        # Calling str(args, kwargs) (line 17)
        str_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 17, 36), str_4, *[x_5], **kwargs_6)
        
        
        # Call to str(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'y' (line 17)
        y_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 50), 'y', False)
        # Processing the call keyword arguments (line 17)
        kwargs_10 = {}
        # Getting the type of 'str' (line 17)
        str_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 46), 'str', False)
        # Calling str(args, kwargs) (line 17)
        str_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 17, 46), str_8, *[y_9], **kwargs_10)
        
        # Applying the binary operator '==' (line 17)
        result_eq_12 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 36), '==', str_call_result_7, str_call_result_11)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 23), 'stypy_return_type', result_eq_12)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 23), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_13

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 23), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 17)
    _stypy_temp_lambda_1_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 23), '_stypy_temp_lambda_1')
    # Processing the call keyword arguments (line 17)
    kwargs_15 = {}
    # Getting the type of 'sorted' (line 17)
    sorted_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'sorted', False)
    # Calling sorted(args, kwargs) (line 17)
    sorted_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), sorted_2, *[list_3, _stypy_temp_lambda_1_14], **kwargs_15)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', sorted_call_result_16)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to sorted(...): (line 19)
    # Processing the call arguments (line 19)
    str_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 17), 'str', 'str')
    # Getting the type of 'types' (line 19)
    types_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 24), 'types', False)
    # Obtaining the member 'FunctionType' of a type (line 19)
    FunctionType_20 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 24), types_19, 'FunctionType')
    # Processing the call keyword arguments (line 19)
    kwargs_21 = {}
    # Getting the type of 'sorted' (line 19)
    sorted_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'sorted', False)
    # Calling sorted(args, kwargs) (line 19)
    sorted_call_result_22 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), sorted_17, *[str_18, FunctionType_20], **kwargs_21)
    
    # Assigning a type to the variable 'ret' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'ret', sorted_call_result_22)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
