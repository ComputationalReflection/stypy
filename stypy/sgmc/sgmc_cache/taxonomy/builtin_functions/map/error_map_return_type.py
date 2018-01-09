
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "map builtin is invoked and its return type is used to call an non existing method"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Has__call__, IterableObject) -> <type 'list'>
7:     # (Has__call__, IterableObject, IterableObject) -> <type 'list'>
8:     # (Has__call__, IterableObject, IterableObject, IterableObject) -> <type 'list'>
9:     # (Has__call__, Str) -> <type 'list'>
10:     # (Has__call__, Str, IterableObject) -> <type 'list'>
11:     # (Has__call__, IterableObject, Str) -> <type 'list'>
12:     # (Has__call__, Str, Str) -> <type 'list'>
13:     # (Has__call__, Str, IterableObject, IterableObject) -> <type 'list'>
14:     # (Has__call__, IterableObject, Str, IterableObject) -> <type 'list'>
15:     # (Has__call__, IterableObject, IterableObject, Str) -> <type 'list'>
16:     # (Has__call__, Str, Str, IterableObject) -> <type 'list'>
17:     # (Has__call__, IterableObject, Str, Str) -> <type 'list'>
18:     # (Has__call__, Str, IterableObject, Str) -> <type 'list'>
19:     # (Has__call__, Str, Str, Str) -> <type 'list'>
20:     # (Has__call__, IterableObject, VarArgs) -> <type 'list'>
21: 
22: 
23:     l = range(5)
24:     l2 = [False, 1, "string"]
25: 
26:     str_l = ["1", "2", "3", "4"]
27: 
28:     other_l = map(lambda x: x, str_l)
29:     # Type error
30:     r1 = other_l[0] + 6
31: 
32: 
33:     def f(x):
34:         return str(x)
35: 
36: 
37:     other_l2 = map(lambda x: f(x), l)
38:     # Type error
39:     r2 = other_l2[0] + 6
40: 
41: 
42:     def f2(x):
43:         if True:
44:             return "3"
45:         else:
46:             return 3
47: 
48: 
49:     other_l3 = map(lambda x: f2(x), l)
50:     # Type warning
51:     x = other_l3[0] + 6
52: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'map builtin is invoked and its return type is used to call an non existing method')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 23):
    
    # Call to range(...): (line 23)
    # Processing the call arguments (line 23)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 14), 'int')
    # Processing the call keyword arguments (line 23)
    kwargs_4 = {}
    # Getting the type of 'range' (line 23)
    range_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'range', False)
    # Calling range(args, kwargs) (line 23)
    range_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), range_2, *[int_3], **kwargs_4)
    
    # Assigning a type to the variable 'l' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'l', range_call_result_5)
    
    # Assigning a List to a Name (line 24):
    
    # Obtaining an instance of the builtin type 'list' (line 24)
    list_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 24)
    # Adding element type (line 24)
    # Getting the type of 'False' (line 24)
    False_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 10), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), list_6, False_7)
    # Adding element type (line 24)
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), list_6, int_8)
    # Adding element type (line 24)
    str_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 20), 'str', 'string')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), list_6, str_9)
    
    # Assigning a type to the variable 'l2' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'l2', list_6)
    
    # Assigning a List to a Name (line 26):
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    str_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 13), 'str', '1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 12), list_10, str_11)
    # Adding element type (line 26)
    str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 18), 'str', '2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 12), list_10, str_12)
    # Adding element type (line 26)
    str_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 23), 'str', '3')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 12), list_10, str_13)
    # Adding element type (line 26)
    str_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 28), 'str', '4')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 12), list_10, str_14)
    
    # Assigning a type to the variable 'str_l' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'str_l', list_10)
    
    # Assigning a Call to a Name (line 28):
    
    # Call to map(...): (line 28)
    # Processing the call arguments (line 28)

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 28, 18, True)
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

        # Getting the type of 'x' (line 28)
        x_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 28), 'x', False)
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), 'stypy_return_type', x_16)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_17

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 28)
    _stypy_temp_lambda_1_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), '_stypy_temp_lambda_1')
    # Getting the type of 'str_l' (line 28)
    str_l_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 31), 'str_l', False)
    # Processing the call keyword arguments (line 28)
    kwargs_20 = {}
    # Getting the type of 'map' (line 28)
    map_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 14), 'map', False)
    # Calling map(args, kwargs) (line 28)
    map_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 28, 14), map_15, *[_stypy_temp_lambda_1_18, str_l_19], **kwargs_20)
    
    # Assigning a type to the variable 'other_l' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'other_l', map_call_result_21)
    
    # Assigning a BinOp to a Name (line 30):
    
    # Obtaining the type of the subscript
    int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 17), 'int')
    # Getting the type of 'other_l' (line 30)
    other_l_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 9), 'other_l')
    # Obtaining the member '__getitem__' of a type (line 30)
    getitem___24 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 9), other_l_23, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 30)
    subscript_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 30, 9), getitem___24, int_22)
    
    int_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 22), 'int')
    # Applying the binary operator '+' (line 30)
    result_add_27 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 9), '+', subscript_call_result_25, int_26)
    
    # Assigning a type to the variable 'r1' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'r1', result_add_27)

    @norecursion
    def f(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 33, 4, False)
        
        # Passed parameters checking function
        f.stypy_localization = localization
        f.stypy_type_of_self = None
        f.stypy_type_store = module_type_store
        f.stypy_function_name = 'f'
        f.stypy_param_names_list = ['x']
        f.stypy_varargs_param_name = None
        f.stypy_kwargs_param_name = None
        f.stypy_call_defaults = defaults
        f.stypy_call_varargs = varargs
        f.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'f', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f(...)' code ##################

        
        # Call to str(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'x' (line 34)
        x_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'x', False)
        # Processing the call keyword arguments (line 34)
        kwargs_30 = {}
        # Getting the type of 'str' (line 34)
        str_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'str', False)
        # Calling str(args, kwargs) (line 34)
        str_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 34, 15), str_28, *[x_29], **kwargs_30)
        
        # Assigning a type to the variable 'stypy_return_type' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'stypy_return_type', str_call_result_31)
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_32)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_32

    # Assigning a type to the variable 'f' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'f', f)
    
    # Assigning a Call to a Name (line 37):
    
    # Call to map(...): (line 37)
    # Processing the call arguments (line 37)

    @norecursion
    def _stypy_temp_lambda_2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_2'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_2', 37, 19, True)
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

        
        # Call to f(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'x' (line 37)
        x_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 31), 'x', False)
        # Processing the call keyword arguments (line 37)
        kwargs_36 = {}
        # Getting the type of 'f' (line 37)
        f_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 29), 'f', False)
        # Calling f(args, kwargs) (line 37)
        f_call_result_37 = invoke(stypy.reporting.localization.Localization(__file__, 37, 29), f_34, *[x_35], **kwargs_36)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 19), 'stypy_return_type', f_call_result_37)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_2' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 19), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_38)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_2'
        return stypy_return_type_38

    # Assigning a type to the variable '_stypy_temp_lambda_2' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 19), '_stypy_temp_lambda_2', _stypy_temp_lambda_2)
    # Getting the type of '_stypy_temp_lambda_2' (line 37)
    _stypy_temp_lambda_2_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 19), '_stypy_temp_lambda_2')
    # Getting the type of 'l' (line 37)
    l_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 35), 'l', False)
    # Processing the call keyword arguments (line 37)
    kwargs_41 = {}
    # Getting the type of 'map' (line 37)
    map_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'map', False)
    # Calling map(args, kwargs) (line 37)
    map_call_result_42 = invoke(stypy.reporting.localization.Localization(__file__, 37, 15), map_33, *[_stypy_temp_lambda_2_39, l_40], **kwargs_41)
    
    # Assigning a type to the variable 'other_l2' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'other_l2', map_call_result_42)
    
    # Assigning a BinOp to a Name (line 39):
    
    # Obtaining the type of the subscript
    int_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 18), 'int')
    # Getting the type of 'other_l2' (line 39)
    other_l2_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 9), 'other_l2')
    # Obtaining the member '__getitem__' of a type (line 39)
    getitem___45 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 9), other_l2_44, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 39)
    subscript_call_result_46 = invoke(stypy.reporting.localization.Localization(__file__, 39, 9), getitem___45, int_43)
    
    int_47 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 23), 'int')
    # Applying the binary operator '+' (line 39)
    result_add_48 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 9), '+', subscript_call_result_46, int_47)
    
    # Assigning a type to the variable 'r2' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'r2', result_add_48)

    @norecursion
    def f2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f2'
        module_type_store = module_type_store.open_function_context('f2', 42, 4, False)
        
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

        
        # Getting the type of 'True' (line 43)
        True_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'True')
        # Testing the type of an if condition (line 43)
        if_condition_50 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 8), True_49)
        # Assigning a type to the variable 'if_condition_50' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'if_condition_50', if_condition_50)
        # SSA begins for if statement (line 43)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_51 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'str', '3')
        # Assigning a type to the variable 'stypy_return_type' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'stypy_return_type', str_51)
        # SSA branch for the else part of an if statement (line 43)
        module_type_store.open_ssa_branch('else')
        int_52 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 19), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'stypy_return_type', int_52)
        # SSA join for if statement (line 43)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'f2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f2' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_53)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f2'
        return stypy_return_type_53

    # Assigning a type to the variable 'f2' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'f2', f2)
    
    # Assigning a Call to a Name (line 49):
    
    # Call to map(...): (line 49)
    # Processing the call arguments (line 49)

    @norecursion
    def _stypy_temp_lambda_3(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_3'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_3', 49, 19, True)
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

        
        # Call to f2(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'x' (line 49)
        x_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 32), 'x', False)
        # Processing the call keyword arguments (line 49)
        kwargs_57 = {}
        # Getting the type of 'f2' (line 49)
        f2_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 29), 'f2', False)
        # Calling f2(args, kwargs) (line 49)
        f2_call_result_58 = invoke(stypy.reporting.localization.Localization(__file__, 49, 29), f2_55, *[x_56], **kwargs_57)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'stypy_return_type', f2_call_result_58)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_3' in the type store
        # Getting the type of 'stypy_return_type' (line 49)
        stypy_return_type_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_3'
        return stypy_return_type_59

    # Assigning a type to the variable '_stypy_temp_lambda_3' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), '_stypy_temp_lambda_3', _stypy_temp_lambda_3)
    # Getting the type of '_stypy_temp_lambda_3' (line 49)
    _stypy_temp_lambda_3_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), '_stypy_temp_lambda_3')
    # Getting the type of 'l' (line 49)
    l_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 36), 'l', False)
    # Processing the call keyword arguments (line 49)
    kwargs_62 = {}
    # Getting the type of 'map' (line 49)
    map_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'map', False)
    # Calling map(args, kwargs) (line 49)
    map_call_result_63 = invoke(stypy.reporting.localization.Localization(__file__, 49, 15), map_54, *[_stypy_temp_lambda_3_60, l_61], **kwargs_62)
    
    # Assigning a type to the variable 'other_l3' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'other_l3', map_call_result_63)
    
    # Assigning a BinOp to a Name (line 51):
    
    # Obtaining the type of the subscript
    int_64 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 17), 'int')
    # Getting the type of 'other_l3' (line 51)
    other_l3_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'other_l3')
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___66 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), other_l3_65, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 51)
    subscript_call_result_67 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), getitem___66, int_64)
    
    int_68 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 22), 'int')
    # Applying the binary operator '+' (line 51)
    result_add_69 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 8), '+', subscript_call_result_67, int_68)
    
    # Assigning a type to the variable 'x' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'x', result_add_69)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
