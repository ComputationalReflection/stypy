
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Checking that the [] operation is applicable to an iterable parameter"
4: 
5: if __name__ == '__main__':
6:     it_list = iter(range(5))
7:     it_tuple = iter(tuple(range(5)))
8:     d = {
9:         "one": 1,
10:         "two": 2,
11:         "three": 3,
12:     }
13: 
14:     it_keys = iter(d.keys())
15:     it_values = iter(d.values())
16:     it_items = iter(d.items())
17: 
18: 
19:     def func(param):
20:         # Type error
21:         print param[3]
22: 
23: 
24:     def func2(param):
25:         # Type error
26:         print param[3]
27: 
28: 
29:     def func3(param):
30:         # Type error
31:         print param[3]
32: 
33: 
34:     def func4(param):
35:         # Type error
36:         print param[3]
37: 
38: 
39:     def func5(param):
40:         # Type error
41:         print param[3]
42: 
43: 
44:     func(it_list)
45:     func2(it_tuple)
46:     func3(it_keys)
47:     func4(it_values)
48:     func5(it_items)
49: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Checking that the [] operation is applicable to an iterable parameter')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 6):
    
    # Call to iter(...): (line 6)
    # Processing the call arguments (line 6)
    
    # Call to range(...): (line 6)
    # Processing the call arguments (line 6)
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 25), 'int')
    # Processing the call keyword arguments (line 6)
    kwargs_5 = {}
    # Getting the type of 'range' (line 6)
    range_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 19), 'range', False)
    # Calling range(args, kwargs) (line 6)
    range_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 6, 19), range_3, *[int_4], **kwargs_5)
    
    # Processing the call keyword arguments (line 6)
    kwargs_7 = {}
    # Getting the type of 'iter' (line 6)
    iter_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 14), 'iter', False)
    # Calling iter(args, kwargs) (line 6)
    iter_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 6, 14), iter_2, *[range_call_result_6], **kwargs_7)
    
    # Assigning a type to the variable 'it_list' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'it_list', iter_call_result_8)
    
    # Assigning a Call to a Name (line 7):
    
    # Call to iter(...): (line 7)
    # Processing the call arguments (line 7)
    
    # Call to tuple(...): (line 7)
    # Processing the call arguments (line 7)
    
    # Call to range(...): (line 7)
    # Processing the call arguments (line 7)
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 32), 'int')
    # Processing the call keyword arguments (line 7)
    kwargs_13 = {}
    # Getting the type of 'range' (line 7)
    range_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 26), 'range', False)
    # Calling range(args, kwargs) (line 7)
    range_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 7, 26), range_11, *[int_12], **kwargs_13)
    
    # Processing the call keyword arguments (line 7)
    kwargs_15 = {}
    # Getting the type of 'tuple' (line 7)
    tuple_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 20), 'tuple', False)
    # Calling tuple(args, kwargs) (line 7)
    tuple_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 7, 20), tuple_10, *[range_call_result_14], **kwargs_15)
    
    # Processing the call keyword arguments (line 7)
    kwargs_17 = {}
    # Getting the type of 'iter' (line 7)
    iter_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 15), 'iter', False)
    # Calling iter(args, kwargs) (line 7)
    iter_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 7, 15), iter_9, *[tuple_call_result_16], **kwargs_17)
    
    # Assigning a type to the variable 'it_tuple' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'it_tuple', iter_call_result_18)
    
    # Assigning a Dict to a Name (line 8):
    
    # Obtaining an instance of the builtin type 'dict' (line 8)
    dict_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 8)
    # Adding element type (key, value) (line 8)
    str_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 8), 'str', 'one')
    int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 8), dict_19, (str_20, int_21))
    # Adding element type (key, value) (line 8)
    str_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'str', 'two')
    int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 15), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 8), dict_19, (str_22, int_23))
    # Adding element type (key, value) (line 8)
    str_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 8), 'str', 'three')
    int_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 17), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 8), dict_19, (str_24, int_25))
    
    # Assigning a type to the variable 'd' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'd', dict_19)
    
    # Assigning a Call to a Name (line 14):
    
    # Call to iter(...): (line 14)
    # Processing the call arguments (line 14)
    
    # Call to keys(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_29 = {}
    # Getting the type of 'd' (line 14)
    d_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'd', False)
    # Obtaining the member 'keys' of a type (line 14)
    keys_28 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 19), d_27, 'keys')
    # Calling keys(args, kwargs) (line 14)
    keys_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 14, 19), keys_28, *[], **kwargs_29)
    
    # Processing the call keyword arguments (line 14)
    kwargs_31 = {}
    # Getting the type of 'iter' (line 14)
    iter_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 14), 'iter', False)
    # Calling iter(args, kwargs) (line 14)
    iter_call_result_32 = invoke(stypy.reporting.localization.Localization(__file__, 14, 14), iter_26, *[keys_call_result_30], **kwargs_31)
    
    # Assigning a type to the variable 'it_keys' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'it_keys', iter_call_result_32)
    
    # Assigning a Call to a Name (line 15):
    
    # Call to iter(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Call to values(...): (line 15)
    # Processing the call keyword arguments (line 15)
    kwargs_36 = {}
    # Getting the type of 'd' (line 15)
    d_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 21), 'd', False)
    # Obtaining the member 'values' of a type (line 15)
    values_35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 21), d_34, 'values')
    # Calling values(args, kwargs) (line 15)
    values_call_result_37 = invoke(stypy.reporting.localization.Localization(__file__, 15, 21), values_35, *[], **kwargs_36)
    
    # Processing the call keyword arguments (line 15)
    kwargs_38 = {}
    # Getting the type of 'iter' (line 15)
    iter_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 16), 'iter', False)
    # Calling iter(args, kwargs) (line 15)
    iter_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 15, 16), iter_33, *[values_call_result_37], **kwargs_38)
    
    # Assigning a type to the variable 'it_values' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'it_values', iter_call_result_39)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to iter(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Call to items(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_43 = {}
    # Getting the type of 'd' (line 16)
    d_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 20), 'd', False)
    # Obtaining the member 'items' of a type (line 16)
    items_42 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 20), d_41, 'items')
    # Calling items(args, kwargs) (line 16)
    items_call_result_44 = invoke(stypy.reporting.localization.Localization(__file__, 16, 20), items_42, *[], **kwargs_43)
    
    # Processing the call keyword arguments (line 16)
    kwargs_45 = {}
    # Getting the type of 'iter' (line 16)
    iter_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'iter', False)
    # Calling iter(args, kwargs) (line 16)
    iter_call_result_46 = invoke(stypy.reporting.localization.Localization(__file__, 16, 15), iter_40, *[items_call_result_44], **kwargs_45)
    
    # Assigning a type to the variable 'it_items' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'it_items', iter_call_result_46)

    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 19, 4, False)
        
        # Passed parameters checking function
        func.stypy_localization = localization
        func.stypy_type_of_self = None
        func.stypy_type_store = module_type_store
        func.stypy_function_name = 'func'
        func.stypy_param_names_list = ['param']
        func.stypy_varargs_param_name = None
        func.stypy_kwargs_param_name = None
        func.stypy_call_defaults = defaults
        func.stypy_call_varargs = varargs
        func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func', ['param'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'func', localization, ['param'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'func(...)' code ##################

        
        # Obtaining the type of the subscript
        int_47 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 20), 'int')
        # Getting the type of 'param' (line 21)
        param_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 14), 'param')
        # Obtaining the member '__getitem__' of a type (line 21)
        getitem___49 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 14), param_48, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 21)
        subscript_call_result_50 = invoke(stypy.reporting.localization.Localization(__file__, 21, 14), getitem___49, int_47)
        
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 19)
        stypy_return_type_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_51)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_51

    # Assigning a type to the variable 'func' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'func', func)

    @norecursion
    def func2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func2'
        module_type_store = module_type_store.open_function_context('func2', 24, 4, False)
        
        # Passed parameters checking function
        func2.stypy_localization = localization
        func2.stypy_type_of_self = None
        func2.stypy_type_store = module_type_store
        func2.stypy_function_name = 'func2'
        func2.stypy_param_names_list = ['param']
        func2.stypy_varargs_param_name = None
        func2.stypy_kwargs_param_name = None
        func2.stypy_call_defaults = defaults
        func2.stypy_call_varargs = varargs
        func2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func2', ['param'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'func2', localization, ['param'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'func2(...)' code ##################

        
        # Obtaining the type of the subscript
        int_52 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 20), 'int')
        # Getting the type of 'param' (line 26)
        param_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'param')
        # Obtaining the member '__getitem__' of a type (line 26)
        getitem___54 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 14), param_53, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 26)
        subscript_call_result_55 = invoke(stypy.reporting.localization.Localization(__file__, 26, 14), getitem___54, int_52)
        
        
        # ################# End of 'func2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func2' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_56)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func2'
        return stypy_return_type_56

    # Assigning a type to the variable 'func2' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'func2', func2)

    @norecursion
    def func3(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func3'
        module_type_store = module_type_store.open_function_context('func3', 29, 4, False)
        
        # Passed parameters checking function
        func3.stypy_localization = localization
        func3.stypy_type_of_self = None
        func3.stypy_type_store = module_type_store
        func3.stypy_function_name = 'func3'
        func3.stypy_param_names_list = ['param']
        func3.stypy_varargs_param_name = None
        func3.stypy_kwargs_param_name = None
        func3.stypy_call_defaults = defaults
        func3.stypy_call_varargs = varargs
        func3.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func3', ['param'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'func3', localization, ['param'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'func3(...)' code ##################

        
        # Obtaining the type of the subscript
        int_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 20), 'int')
        # Getting the type of 'param' (line 31)
        param_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'param')
        # Obtaining the member '__getitem__' of a type (line 31)
        getitem___59 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 14), param_58, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 31)
        subscript_call_result_60 = invoke(stypy.reporting.localization.Localization(__file__, 31, 14), getitem___59, int_57)
        
        
        # ################# End of 'func3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func3' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func3'
        return stypy_return_type_61

    # Assigning a type to the variable 'func3' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'func3', func3)

    @norecursion
    def func4(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func4'
        module_type_store = module_type_store.open_function_context('func4', 34, 4, False)
        
        # Passed parameters checking function
        func4.stypy_localization = localization
        func4.stypy_type_of_self = None
        func4.stypy_type_store = module_type_store
        func4.stypy_function_name = 'func4'
        func4.stypy_param_names_list = ['param']
        func4.stypy_varargs_param_name = None
        func4.stypy_kwargs_param_name = None
        func4.stypy_call_defaults = defaults
        func4.stypy_call_varargs = varargs
        func4.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func4', ['param'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'func4', localization, ['param'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'func4(...)' code ##################

        
        # Obtaining the type of the subscript
        int_62 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 20), 'int')
        # Getting the type of 'param' (line 36)
        param_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 14), 'param')
        # Obtaining the member '__getitem__' of a type (line 36)
        getitem___64 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 14), param_63, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 36)
        subscript_call_result_65 = invoke(stypy.reporting.localization.Localization(__file__, 36, 14), getitem___64, int_62)
        
        
        # ################# End of 'func4(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func4' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_66)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func4'
        return stypy_return_type_66

    # Assigning a type to the variable 'func4' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'func4', func4)

    @norecursion
    def func5(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func5'
        module_type_store = module_type_store.open_function_context('func5', 39, 4, False)
        
        # Passed parameters checking function
        func5.stypy_localization = localization
        func5.stypy_type_of_self = None
        func5.stypy_type_store = module_type_store
        func5.stypy_function_name = 'func5'
        func5.stypy_param_names_list = ['param']
        func5.stypy_varargs_param_name = None
        func5.stypy_kwargs_param_name = None
        func5.stypy_call_defaults = defaults
        func5.stypy_call_varargs = varargs
        func5.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func5', ['param'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'func5', localization, ['param'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'func5(...)' code ##################

        
        # Obtaining the type of the subscript
        int_67 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 20), 'int')
        # Getting the type of 'param' (line 41)
        param_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 14), 'param')
        # Obtaining the member '__getitem__' of a type (line 41)
        getitem___69 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 14), param_68, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 41)
        subscript_call_result_70 = invoke(stypy.reporting.localization.Localization(__file__, 41, 14), getitem___69, int_67)
        
        
        # ################# End of 'func5(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func5' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_71)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func5'
        return stypy_return_type_71

    # Assigning a type to the variable 'func5' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'func5', func5)
    
    # Call to func(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'it_list' (line 44)
    it_list_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 9), 'it_list', False)
    # Processing the call keyword arguments (line 44)
    kwargs_74 = {}
    # Getting the type of 'func' (line 44)
    func_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'func', False)
    # Calling func(args, kwargs) (line 44)
    func_call_result_75 = invoke(stypy.reporting.localization.Localization(__file__, 44, 4), func_72, *[it_list_73], **kwargs_74)
    
    
    # Call to func2(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'it_tuple' (line 45)
    it_tuple_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 10), 'it_tuple', False)
    # Processing the call keyword arguments (line 45)
    kwargs_78 = {}
    # Getting the type of 'func2' (line 45)
    func2_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'func2', False)
    # Calling func2(args, kwargs) (line 45)
    func2_call_result_79 = invoke(stypy.reporting.localization.Localization(__file__, 45, 4), func2_76, *[it_tuple_77], **kwargs_78)
    
    
    # Call to func3(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'it_keys' (line 46)
    it_keys_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 10), 'it_keys', False)
    # Processing the call keyword arguments (line 46)
    kwargs_82 = {}
    # Getting the type of 'func3' (line 46)
    func3_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'func3', False)
    # Calling func3(args, kwargs) (line 46)
    func3_call_result_83 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), func3_80, *[it_keys_81], **kwargs_82)
    
    
    # Call to func4(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'it_values' (line 47)
    it_values_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 10), 'it_values', False)
    # Processing the call keyword arguments (line 47)
    kwargs_86 = {}
    # Getting the type of 'func4' (line 47)
    func4_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'func4', False)
    # Calling func4(args, kwargs) (line 47)
    func4_call_result_87 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), func4_84, *[it_values_85], **kwargs_86)
    
    
    # Call to func5(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'it_items' (line 48)
    it_items_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 10), 'it_items', False)
    # Processing the call keyword arguments (line 48)
    kwargs_90 = {}
    # Getting the type of 'func5' (line 48)
    func5_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'func5', False)
    # Calling func5(args, kwargs) (line 48)
    func5_call_result_91 = invoke(stypy.reporting.localization.Localization(__file__, 48, 4), func5_88, *[it_items_89], **kwargs_90)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
