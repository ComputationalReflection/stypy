
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "filter method is present, but is invoked with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Has__call__, IterableObject) -> DynamicType
7:     # (Has__call__, Str) -> DynamicType
8: 
9: 
10:     # Call the builtin with incorrect number of parameters
11:     l2 = [False, 1, "string"]
12: 
13: 
14:     def f2(x):
15:         return str(x)
16: 
17: 
18:     # Type error
19:     other_l = filter(lambda x: f2(x), l2, l2)
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'filter method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a List to a Name (line 11):
    
    # Obtaining an instance of the builtin type 'list' (line 11)
    list_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 11)
    # Adding element type (line 11)
    # Getting the type of 'False' (line 11)
    False_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 9), list_2, False_3)
    # Adding element type (line 11)
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 9), list_2, int_4)
    # Adding element type (line 11)
    str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 20), 'str', 'string')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 9), list_2, str_5)
    
    # Assigning a type to the variable 'l2' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'l2', list_2)

    @norecursion
    def f2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f2'
        module_type_store = module_type_store.open_function_context('f2', 14, 4, False)
        
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

        
        # Call to str(...): (line 15)
        # Processing the call arguments (line 15)
        # Getting the type of 'x' (line 15)
        x_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 19), 'x', False)
        # Processing the call keyword arguments (line 15)
        kwargs_8 = {}
        # Getting the type of 'str' (line 15)
        str_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', False)
        # Calling str(args, kwargs) (line 15)
        str_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 15, 15), str_6, *[x_7], **kwargs_8)
        
        # Assigning a type to the variable 'stypy_return_type' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', str_call_result_9)
        
        # ################# End of 'f2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f2' in the type store
        # Getting the type of 'stypy_return_type' (line 14)
        stypy_return_type_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f2'
        return stypy_return_type_10

    # Assigning a type to the variable 'f2' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'f2', f2)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to filter(...): (line 19)
    # Processing the call arguments (line 19)

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 19, 21, True)
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

        
        # Call to f2(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'x' (line 19)
        x_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 34), 'x', False)
        # Processing the call keyword arguments (line 19)
        kwargs_14 = {}
        # Getting the type of 'f2' (line 19)
        f2_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 31), 'f2', False)
        # Calling f2(args, kwargs) (line 19)
        f2_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 19, 31), f2_12, *[x_13], **kwargs_14)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 21), 'stypy_return_type', f2_call_result_15)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 19)
        stypy_return_type_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 21), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_16

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 21), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 19)
    _stypy_temp_lambda_1_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 21), '_stypy_temp_lambda_1')
    # Getting the type of 'l2' (line 19)
    l2_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 38), 'l2', False)
    # Getting the type of 'l2' (line 19)
    l2_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 42), 'l2', False)
    # Processing the call keyword arguments (line 19)
    kwargs_20 = {}
    # Getting the type of 'filter' (line 19)
    filter_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 14), 'filter', False)
    # Calling filter(args, kwargs) (line 19)
    filter_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 19, 14), filter_11, *[_stypy_temp_lambda_1_17, l2_18, l2_19], **kwargs_20)
    
    # Assigning a type to the variable 'other_l' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'other_l', filter_call_result_21)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
