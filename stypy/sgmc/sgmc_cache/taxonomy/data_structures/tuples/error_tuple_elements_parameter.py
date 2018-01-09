
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Inferring the type of the element when the tuple is passed as a parameter"
4: 
5: if __name__ == '__main__':
6:     def fun1(l):
7:         # Type error
8:         return "aaa" + l[0]
9: 
10: 
11:     def fun1b(l):
12:         # Type error
13:         return "aaa" + l[0]
14: 
15: 
16:     def fun1c(l):
17:         # Type error
18:         return "aaa" + l[0]
19: 
20: 
21:     def fun1d(l):
22:         # Type error
23:         return "aaa" + l[0]
24: 
25: 
26:     def fun2(l):
27:         # Type error
28:         return 3 / l[0]
29: 
30: 
31:     tuple_ = (3, 4)
32:     S = tuple([x ** 2 for x in range(10)])
33:     V = tuple([str(i) for i in range(13)])
34:     normal_list = tuple([1, 2, 3])
35: 
36:     r = fun1(S[0])
37:     r2 = fun1b(S)
38:     r3 = fun1c(normal_list)
39:     r4 = fun1d(tuple_)
40:     r5 = fun2(V)
41: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Inferring the type of the element when the tuple is passed as a parameter')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def fun1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun1'
        module_type_store = module_type_store.open_function_context('fun1', 6, 4, False)
        
        # Passed parameters checking function
        fun1.stypy_localization = localization
        fun1.stypy_type_of_self = None
        fun1.stypy_type_store = module_type_store
        fun1.stypy_function_name = 'fun1'
        fun1.stypy_param_names_list = ['l']
        fun1.stypy_varargs_param_name = None
        fun1.stypy_kwargs_param_name = None
        fun1.stypy_call_defaults = defaults
        fun1.stypy_call_varargs = varargs
        fun1.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'fun1', ['l'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun1', localization, ['l'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun1(...)' code ##################

        str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 15), 'str', 'aaa')
        
        # Obtaining the type of the subscript
        int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 25), 'int')
        # Getting the type of 'l' (line 8)
        l_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 23), 'l')
        # Obtaining the member '__getitem__' of a type (line 8)
        getitem___5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 23), l_4, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 8)
        subscript_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 8, 23), getitem___5, int_3)
        
        # Applying the binary operator '+' (line 8)
        result_add_7 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 15), '+', str_2, subscript_call_result_6)
        
        # Assigning a type to the variable 'stypy_return_type' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type', result_add_7)
        
        # ################# End of 'fun1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun1' in the type store
        # Getting the type of 'stypy_return_type' (line 6)
        stypy_return_type_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun1'
        return stypy_return_type_8

    # Assigning a type to the variable 'fun1' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'fun1', fun1)

    @norecursion
    def fun1b(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun1b'
        module_type_store = module_type_store.open_function_context('fun1b', 11, 4, False)
        
        # Passed parameters checking function
        fun1b.stypy_localization = localization
        fun1b.stypy_type_of_self = None
        fun1b.stypy_type_store = module_type_store
        fun1b.stypy_function_name = 'fun1b'
        fun1b.stypy_param_names_list = ['l']
        fun1b.stypy_varargs_param_name = None
        fun1b.stypy_kwargs_param_name = None
        fun1b.stypy_call_defaults = defaults
        fun1b.stypy_call_varargs = varargs
        fun1b.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'fun1b', ['l'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun1b', localization, ['l'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun1b(...)' code ##################

        str_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'aaa')
        
        # Obtaining the type of the subscript
        int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 25), 'int')
        # Getting the type of 'l' (line 13)
        l_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 23), 'l')
        # Obtaining the member '__getitem__' of a type (line 13)
        getitem___12 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 23), l_11, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 13)
        subscript_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 13, 23), getitem___12, int_10)
        
        # Applying the binary operator '+' (line 13)
        result_add_14 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 15), '+', str_9, subscript_call_result_13)
        
        # Assigning a type to the variable 'stypy_return_type' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', result_add_14)
        
        # ################# End of 'fun1b(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun1b' in the type store
        # Getting the type of 'stypy_return_type' (line 11)
        stypy_return_type_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun1b'
        return stypy_return_type_15

    # Assigning a type to the variable 'fun1b' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'fun1b', fun1b)

    @norecursion
    def fun1c(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun1c'
        module_type_store = module_type_store.open_function_context('fun1c', 16, 4, False)
        
        # Passed parameters checking function
        fun1c.stypy_localization = localization
        fun1c.stypy_type_of_self = None
        fun1c.stypy_type_store = module_type_store
        fun1c.stypy_function_name = 'fun1c'
        fun1c.stypy_param_names_list = ['l']
        fun1c.stypy_varargs_param_name = None
        fun1c.stypy_kwargs_param_name = None
        fun1c.stypy_call_defaults = defaults
        fun1c.stypy_call_varargs = varargs
        fun1c.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'fun1c', ['l'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun1c', localization, ['l'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun1c(...)' code ##################

        str_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 15), 'str', 'aaa')
        
        # Obtaining the type of the subscript
        int_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'int')
        # Getting the type of 'l' (line 18)
        l_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 23), 'l')
        # Obtaining the member '__getitem__' of a type (line 18)
        getitem___19 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 23), l_18, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 18)
        subscript_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 18, 23), getitem___19, int_17)
        
        # Applying the binary operator '+' (line 18)
        result_add_21 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 15), '+', str_16, subscript_call_result_20)
        
        # Assigning a type to the variable 'stypy_return_type' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'stypy_return_type', result_add_21)
        
        # ################# End of 'fun1c(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun1c' in the type store
        # Getting the type of 'stypy_return_type' (line 16)
        stypy_return_type_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_22)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun1c'
        return stypy_return_type_22

    # Assigning a type to the variable 'fun1c' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'fun1c', fun1c)

    @norecursion
    def fun1d(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun1d'
        module_type_store = module_type_store.open_function_context('fun1d', 21, 4, False)
        
        # Passed parameters checking function
        fun1d.stypy_localization = localization
        fun1d.stypy_type_of_self = None
        fun1d.stypy_type_store = module_type_store
        fun1d.stypy_function_name = 'fun1d'
        fun1d.stypy_param_names_list = ['l']
        fun1d.stypy_varargs_param_name = None
        fun1d.stypy_kwargs_param_name = None
        fun1d.stypy_call_defaults = defaults
        fun1d.stypy_call_varargs = varargs
        fun1d.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'fun1d', ['l'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun1d', localization, ['l'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun1d(...)' code ##################

        str_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'str', 'aaa')
        
        # Obtaining the type of the subscript
        int_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 25), 'int')
        # Getting the type of 'l' (line 23)
        l_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 23), 'l')
        # Obtaining the member '__getitem__' of a type (line 23)
        getitem___26 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 23), l_25, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 23)
        subscript_call_result_27 = invoke(stypy.reporting.localization.Localization(__file__, 23, 23), getitem___26, int_24)
        
        # Applying the binary operator '+' (line 23)
        result_add_28 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 15), '+', str_23, subscript_call_result_27)
        
        # Assigning a type to the variable 'stypy_return_type' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'stypy_return_type', result_add_28)
        
        # ################# End of 'fun1d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun1d' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun1d'
        return stypy_return_type_29

    # Assigning a type to the variable 'fun1d' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'fun1d', fun1d)

    @norecursion
    def fun2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun2'
        module_type_store = module_type_store.open_function_context('fun2', 26, 4, False)
        
        # Passed parameters checking function
        fun2.stypy_localization = localization
        fun2.stypy_type_of_self = None
        fun2.stypy_type_store = module_type_store
        fun2.stypy_function_name = 'fun2'
        fun2.stypy_param_names_list = ['l']
        fun2.stypy_varargs_param_name = None
        fun2.stypy_kwargs_param_name = None
        fun2.stypy_call_defaults = defaults
        fun2.stypy_call_varargs = varargs
        fun2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'fun2', ['l'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun2', localization, ['l'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun2(...)' code ##################

        int_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 15), 'int')
        
        # Obtaining the type of the subscript
        int_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 21), 'int')
        # Getting the type of 'l' (line 28)
        l_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 19), 'l')
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___33 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 19), l_32, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 28)
        subscript_call_result_34 = invoke(stypy.reporting.localization.Localization(__file__, 28, 19), getitem___33, int_31)
        
        # Applying the binary operator 'div' (line 28)
        result_div_35 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 15), 'div', int_30, subscript_call_result_34)
        
        # Assigning a type to the variable 'stypy_return_type' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stypy_return_type', result_div_35)
        
        # ################# End of 'fun2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun2' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun2'
        return stypy_return_type_36

    # Assigning a type to the variable 'fun2' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'fun2', fun2)
    
    # Assigning a Tuple to a Name (line 31):
    
    # Obtaining an instance of the builtin type 'tuple' (line 31)
    tuple_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 31)
    # Adding element type (line 31)
    int_38 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 14), tuple_37, int_38)
    # Adding element type (line 31)
    int_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 14), tuple_37, int_39)
    
    # Assigning a type to the variable 'tuple_' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'tuple_', tuple_37)
    
    # Assigning a Call to a Name (line 32):
    
    # Call to tuple(...): (line 32)
    # Processing the call arguments (line 32)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 32)
    # Processing the call arguments (line 32)
    int_45 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 37), 'int')
    # Processing the call keyword arguments (line 32)
    kwargs_46 = {}
    # Getting the type of 'range' (line 32)
    range_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 31), 'range', False)
    # Calling range(args, kwargs) (line 32)
    range_call_result_47 = invoke(stypy.reporting.localization.Localization(__file__, 32, 31), range_44, *[int_45], **kwargs_46)
    
    comprehension_48 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 15), range_call_result_47)
    # Assigning a type to the variable 'x' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'x', comprehension_48)
    # Getting the type of 'x' (line 32)
    x_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'x', False)
    int_42 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 20), 'int')
    # Applying the binary operator '**' (line 32)
    result_pow_43 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 15), '**', x_41, int_42)
    
    list_49 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 15), list_49, result_pow_43)
    # Processing the call keyword arguments (line 32)
    kwargs_50 = {}
    # Getting the type of 'tuple' (line 32)
    tuple_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'tuple', False)
    # Calling tuple(args, kwargs) (line 32)
    tuple_call_result_51 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), tuple_40, *[list_49], **kwargs_50)
    
    # Assigning a type to the variable 'S' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'S', tuple_call_result_51)
    
    # Assigning a Call to a Name (line 33):
    
    # Call to tuple(...): (line 33)
    # Processing the call arguments (line 33)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 33)
    # Processing the call arguments (line 33)
    int_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 37), 'int')
    # Processing the call keyword arguments (line 33)
    kwargs_59 = {}
    # Getting the type of 'range' (line 33)
    range_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 31), 'range', False)
    # Calling range(args, kwargs) (line 33)
    range_call_result_60 = invoke(stypy.reporting.localization.Localization(__file__, 33, 31), range_57, *[int_58], **kwargs_59)
    
    comprehension_61 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 15), range_call_result_60)
    # Assigning a type to the variable 'i' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'i', comprehension_61)
    
    # Call to str(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'i' (line 33)
    i_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'i', False)
    # Processing the call keyword arguments (line 33)
    kwargs_55 = {}
    # Getting the type of 'str' (line 33)
    str_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'str', False)
    # Calling str(args, kwargs) (line 33)
    str_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 33, 15), str_53, *[i_54], **kwargs_55)
    
    list_62 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 15), list_62, str_call_result_56)
    # Processing the call keyword arguments (line 33)
    kwargs_63 = {}
    # Getting the type of 'tuple' (line 33)
    tuple_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'tuple', False)
    # Calling tuple(args, kwargs) (line 33)
    tuple_call_result_64 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), tuple_52, *[list_62], **kwargs_63)
    
    # Assigning a type to the variable 'V' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'V', tuple_call_result_64)
    
    # Assigning a Call to a Name (line 34):
    
    # Call to tuple(...): (line 34)
    # Processing the call arguments (line 34)
    
    # Obtaining an instance of the builtin type 'list' (line 34)
    list_66 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 34)
    # Adding element type (line 34)
    int_67 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 24), list_66, int_67)
    # Adding element type (line 34)
    int_68 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 24), list_66, int_68)
    # Adding element type (line 34)
    int_69 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 24), list_66, int_69)
    
    # Processing the call keyword arguments (line 34)
    kwargs_70 = {}
    # Getting the type of 'tuple' (line 34)
    tuple_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 18), 'tuple', False)
    # Calling tuple(args, kwargs) (line 34)
    tuple_call_result_71 = invoke(stypy.reporting.localization.Localization(__file__, 34, 18), tuple_65, *[list_66], **kwargs_70)
    
    # Assigning a type to the variable 'normal_list' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'normal_list', tuple_call_result_71)
    
    # Assigning a Call to a Name (line 36):
    
    # Call to fun1(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Obtaining the type of the subscript
    int_73 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 15), 'int')
    # Getting the type of 'S' (line 36)
    S_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 13), 'S', False)
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___75 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 13), S_74, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_76 = invoke(stypy.reporting.localization.Localization(__file__, 36, 13), getitem___75, int_73)
    
    # Processing the call keyword arguments (line 36)
    kwargs_77 = {}
    # Getting the type of 'fun1' (line 36)
    fun1_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'fun1', False)
    # Calling fun1(args, kwargs) (line 36)
    fun1_call_result_78 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), fun1_72, *[subscript_call_result_76], **kwargs_77)
    
    # Assigning a type to the variable 'r' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'r', fun1_call_result_78)
    
    # Assigning a Call to a Name (line 37):
    
    # Call to fun1b(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'S' (line 37)
    S_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'S', False)
    # Processing the call keyword arguments (line 37)
    kwargs_81 = {}
    # Getting the type of 'fun1b' (line 37)
    fun1b_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 9), 'fun1b', False)
    # Calling fun1b(args, kwargs) (line 37)
    fun1b_call_result_82 = invoke(stypy.reporting.localization.Localization(__file__, 37, 9), fun1b_79, *[S_80], **kwargs_81)
    
    # Assigning a type to the variable 'r2' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'r2', fun1b_call_result_82)
    
    # Assigning a Call to a Name (line 38):
    
    # Call to fun1c(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'normal_list' (line 38)
    normal_list_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'normal_list', False)
    # Processing the call keyword arguments (line 38)
    kwargs_85 = {}
    # Getting the type of 'fun1c' (line 38)
    fun1c_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 9), 'fun1c', False)
    # Calling fun1c(args, kwargs) (line 38)
    fun1c_call_result_86 = invoke(stypy.reporting.localization.Localization(__file__, 38, 9), fun1c_83, *[normal_list_84], **kwargs_85)
    
    # Assigning a type to the variable 'r3' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'r3', fun1c_call_result_86)
    
    # Assigning a Call to a Name (line 39):
    
    # Call to fun1d(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'tuple_' (line 39)
    tuple__88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'tuple_', False)
    # Processing the call keyword arguments (line 39)
    kwargs_89 = {}
    # Getting the type of 'fun1d' (line 39)
    fun1d_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 9), 'fun1d', False)
    # Calling fun1d(args, kwargs) (line 39)
    fun1d_call_result_90 = invoke(stypy.reporting.localization.Localization(__file__, 39, 9), fun1d_87, *[tuple__88], **kwargs_89)
    
    # Assigning a type to the variable 'r4' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'r4', fun1d_call_result_90)
    
    # Assigning a Call to a Name (line 40):
    
    # Call to fun2(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'V' (line 40)
    V_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 14), 'V', False)
    # Processing the call keyword arguments (line 40)
    kwargs_93 = {}
    # Getting the type of 'fun2' (line 40)
    fun2_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 9), 'fun2', False)
    # Calling fun2(args, kwargs) (line 40)
    fun2_call_result_94 = invoke(stypy.reporting.localization.Localization(__file__, 40, 9), fun2_91, *[V_92], **kwargs_93)
    
    # Assigning a type to the variable 'r5' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'r5', fun2_call_result_94)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
