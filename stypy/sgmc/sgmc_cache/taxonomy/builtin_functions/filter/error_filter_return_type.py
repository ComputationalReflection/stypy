
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "filter builtin is invoked and its return type is used to call an non existing method"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Has__call__, IterableObject) -> DynamicType
7:     # (Has__call__, Str) -> DynamicType
8: 
9: 
10:     l2 = [False, 1, "string"]
11: 
12: 
13:     def f2(x):
14:         return str(x)
15: 
16: 
17:     # No error
18:     other_l = filter(lambda x: f2(x), l2)
19: 
20:     # Type warning
21:     r1 = other_l[2] + 6
22: 
23:     l3 = ["False", "1", "string"]
24:     # No error
25:     other_l2 = filter(lambda x: f2(x), l3)
26: 
27:     # Type error
28:     r2 = other_l2[2] + 6
29: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'filter builtin is invoked and its return type is used to call an non existing method')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a List to a Name (line 10):
    
    # Obtaining an instance of the builtin type 'list' (line 10)
    list_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 10)
    # Adding element type (line 10)
    # Getting the type of 'False' (line 10)
    False_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 10), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 9), list_2, False_3)
    # Adding element type (line 10)
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 9), list_2, int_4)
    # Adding element type (line 10)
    str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 20), 'str', 'string')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 9), list_2, str_5)
    
    # Assigning a type to the variable 'l2' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'l2', list_2)

    @norecursion
    def f2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f2'
        module_type_store = module_type_store.open_function_context('f2', 13, 4, False)
        
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

        
        # Call to str(...): (line 14)
        # Processing the call arguments (line 14)
        # Getting the type of 'x' (line 14)
        x_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'x', False)
        # Processing the call keyword arguments (line 14)
        kwargs_8 = {}
        # Getting the type of 'str' (line 14)
        str_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'str', False)
        # Calling str(args, kwargs) (line 14)
        str_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 14, 15), str_6, *[x_7], **kwargs_8)
        
        # Assigning a type to the variable 'stypy_return_type' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'stypy_return_type', str_call_result_9)
        
        # ################# End of 'f2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f2' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f2'
        return stypy_return_type_10

    # Assigning a type to the variable 'f2' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'f2', f2)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to filter(...): (line 18)
    # Processing the call arguments (line 18)

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 18, 21, True)
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

        
        # Call to f2(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'x' (line 18)
        x_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 34), 'x', False)
        # Processing the call keyword arguments (line 18)
        kwargs_14 = {}
        # Getting the type of 'f2' (line 18)
        f2_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 31), 'f2', False)
        # Calling f2(args, kwargs) (line 18)
        f2_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 18, 31), f2_12, *[x_13], **kwargs_14)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 21), 'stypy_return_type', f2_call_result_15)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 21), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_16

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 21), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 18)
    _stypy_temp_lambda_1_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 21), '_stypy_temp_lambda_1')
    # Getting the type of 'l2' (line 18)
    l2_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 38), 'l2', False)
    # Processing the call keyword arguments (line 18)
    kwargs_19 = {}
    # Getting the type of 'filter' (line 18)
    filter_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), 'filter', False)
    # Calling filter(args, kwargs) (line 18)
    filter_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 18, 14), filter_11, *[_stypy_temp_lambda_1_17, l2_18], **kwargs_19)
    
    # Assigning a type to the variable 'other_l' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'other_l', filter_call_result_20)
    
    # Assigning a BinOp to a Name (line 21):
    
    # Obtaining the type of the subscript
    int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 17), 'int')
    # Getting the type of 'other_l' (line 21)
    other_l_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 9), 'other_l')
    # Obtaining the member '__getitem__' of a type (line 21)
    getitem___23 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 9), other_l_22, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 21)
    subscript_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 21, 9), getitem___23, int_21)
    
    int_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 22), 'int')
    # Applying the binary operator '+' (line 21)
    result_add_26 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 9), '+', subscript_call_result_24, int_25)
    
    # Assigning a type to the variable 'r1' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'r1', result_add_26)
    
    # Assigning a List to a Name (line 23):
    
    # Obtaining an instance of the builtin type 'list' (line 23)
    list_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 23)
    # Adding element type (line 23)
    str_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 10), 'str', 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), list_27, str_28)
    # Adding element type (line 23)
    str_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 19), 'str', '1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), list_27, str_29)
    # Adding element type (line 23)
    str_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 24), 'str', 'string')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), list_27, str_30)
    
    # Assigning a type to the variable 'l3' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'l3', list_27)
    
    # Assigning a Call to a Name (line 25):
    
    # Call to filter(...): (line 25)
    # Processing the call arguments (line 25)

    @norecursion
    def _stypy_temp_lambda_2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_2'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_2', 25, 22, True)
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

        
        # Call to f2(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'x' (line 25)
        x_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 35), 'x', False)
        # Processing the call keyword arguments (line 25)
        kwargs_34 = {}
        # Getting the type of 'f2' (line 25)
        f2_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 32), 'f2', False)
        # Calling f2(args, kwargs) (line 25)
        f2_call_result_35 = invoke(stypy.reporting.localization.Localization(__file__, 25, 32), f2_32, *[x_33], **kwargs_34)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 22), 'stypy_return_type', f2_call_result_35)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_2' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 22), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_2'
        return stypy_return_type_36

    # Assigning a type to the variable '_stypy_temp_lambda_2' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 22), '_stypy_temp_lambda_2', _stypy_temp_lambda_2)
    # Getting the type of '_stypy_temp_lambda_2' (line 25)
    _stypy_temp_lambda_2_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 22), '_stypy_temp_lambda_2')
    # Getting the type of 'l3' (line 25)
    l3_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 39), 'l3', False)
    # Processing the call keyword arguments (line 25)
    kwargs_39 = {}
    # Getting the type of 'filter' (line 25)
    filter_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'filter', False)
    # Calling filter(args, kwargs) (line 25)
    filter_call_result_40 = invoke(stypy.reporting.localization.Localization(__file__, 25, 15), filter_31, *[_stypy_temp_lambda_2_37, l3_38], **kwargs_39)
    
    # Assigning a type to the variable 'other_l2' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'other_l2', filter_call_result_40)
    
    # Assigning a BinOp to a Name (line 28):
    
    # Obtaining the type of the subscript
    int_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 18), 'int')
    # Getting the type of 'other_l2' (line 28)
    other_l2_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 9), 'other_l2')
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___43 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 9), other_l2_42, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_44 = invoke(stypy.reporting.localization.Localization(__file__, 28, 9), getitem___43, int_41)
    
    int_45 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 23), 'int')
    # Applying the binary operator '+' (line 28)
    result_add_46 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 9), '+', subscript_call_result_44, int_45)
    
    # Assigning a type to the variable 'r2' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'r2', result_add_46)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
