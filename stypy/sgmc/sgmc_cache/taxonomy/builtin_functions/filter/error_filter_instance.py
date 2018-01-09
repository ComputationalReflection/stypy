
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "filter builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Has__call__, IterableObject) -> DynamicType
7:     # (Has__call__, Str) -> DynamicType
8: 
9:     import types
10: 
11: 
12:     def f2(x):
13:         return str(x)
14: 
15: 
16:     # Type error
17:     other_l3 = filter(lambda x: f2(x), list)
18:     # Type error
19:     other_l3 = filter(types.FunctionType, list)
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'filter builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 4))
    
    # 'import types' statement (line 9)
    import types

    import_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'types', types, module_type_store)
    

    @norecursion
    def f2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f2'
        module_type_store = module_type_store.open_function_context('f2', 12, 4, False)
        
        # Passed parameters checking function
        f2.stypy_localization = localization
        f2.stypy_type_of_self = None
        f2.stypy_type_store = module_type_store
        f2.stypy_function_name = 'f2'
        f2.stypy_param_names_list = ['x']
        f2.stypy_varargs_param_name = None
        f2.stypy_kwargs_param_name = None
        f2.stypy_call_defaults = defaults
        f2.stypy_call_varargs = varargs
        f2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'f2', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f2', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f2(...)' code ##################

        
        # Call to str(...): (line 13)
        # Processing the call arguments (line 13)
        # Getting the type of 'x' (line 13)
        x_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 19), 'x', False)
        # Processing the call keyword arguments (line 13)
        kwargs_4 = {}
        # Getting the type of 'str' (line 13)
        str_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', False)
        # Calling str(args, kwargs) (line 13)
        str_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 13, 15), str_2, *[x_3], **kwargs_4)
        
        # Assigning a type to the variable 'stypy_return_type' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', str_call_result_5)
        
        # ################# End of 'f2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f2' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f2'
        return stypy_return_type_6

    # Assigning a type to the variable 'f2' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'f2', f2)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to filter(...): (line 17)
    # Processing the call arguments (line 17)

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 17, 22, True)
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

        
        # Call to f2(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'x' (line 17)
        x_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 35), 'x', False)
        # Processing the call keyword arguments (line 17)
        kwargs_10 = {}
        # Getting the type of 'f2' (line 17)
        f2_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 32), 'f2', False)
        # Calling f2(args, kwargs) (line 17)
        f2_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 17, 32), f2_8, *[x_9], **kwargs_10)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 22), 'stypy_return_type', f2_call_result_11)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 22), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_12

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 22), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 17)
    _stypy_temp_lambda_1_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 22), '_stypy_temp_lambda_1')
    # Getting the type of 'list' (line 17)
    list_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 39), 'list', False)
    # Processing the call keyword arguments (line 17)
    kwargs_15 = {}
    # Getting the type of 'filter' (line 17)
    filter_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'filter', False)
    # Calling filter(args, kwargs) (line 17)
    filter_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 17, 15), filter_7, *[_stypy_temp_lambda_1_13, list_14], **kwargs_15)
    
    # Assigning a type to the variable 'other_l3' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'other_l3', filter_call_result_16)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to filter(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'types' (line 19)
    types_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'types', False)
    # Obtaining the member 'FunctionType' of a type (line 19)
    FunctionType_19 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 22), types_18, 'FunctionType')
    # Getting the type of 'list' (line 19)
    list_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 42), 'list', False)
    # Processing the call keyword arguments (line 19)
    kwargs_21 = {}
    # Getting the type of 'filter' (line 19)
    filter_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'filter', False)
    # Calling filter(args, kwargs) (line 19)
    filter_call_result_22 = invoke(stypy.reporting.localization.Localization(__file__, 19, 15), filter_17, *[FunctionType_19, list_20], **kwargs_21)
    
    # Assigning a type to the variable 'other_l3' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'other_l3', filter_call_result_22)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
