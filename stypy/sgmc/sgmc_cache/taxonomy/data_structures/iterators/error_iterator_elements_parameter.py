
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Inferring the type of the element when the iterator is passed as a parameter"
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
21:     def fun2(l):
22:         # Type error
23:         return 3 / l[0]
24: 
25: 
26:     tuple_ = iter((3, 4))
27:     S = iter([x ** 2 for x in range(10)])
28:     V = iter([str(i) for i in range(13)])
29:     normal_list = iter([1, 2, 3])
30: 
31:     r2 = fun1(S)
32:     r3 = fun1b(normal_list)
33:     r4 = fun1c(tuple_)
34:     r5 = fun2(V)
35: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Inferring the type of the element when the iterator is passed as a parameter')
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
    def fun2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun2'
        module_type_store = module_type_store.open_function_context('fun2', 21, 4, False)
        
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

        int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'int')
        
        # Obtaining the type of the subscript
        int_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'int')
        # Getting the type of 'l' (line 23)
        l_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'l')
        # Obtaining the member '__getitem__' of a type (line 23)
        getitem___26 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 19), l_25, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 23)
        subscript_call_result_27 = invoke(stypy.reporting.localization.Localization(__file__, 23, 19), getitem___26, int_24)
        
        # Applying the binary operator 'div' (line 23)
        result_div_28 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 15), 'div', int_23, subscript_call_result_27)
        
        # Assigning a type to the variable 'stypy_return_type' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'stypy_return_type', result_div_28)
        
        # ################# End of 'fun2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun2' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_29)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun2'
        return stypy_return_type_29

    # Assigning a type to the variable 'fun2' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'fun2', fun2)
    
    # Assigning a Call to a Name (line 26):
    
    # Call to iter(...): (line 26)
    # Processing the call arguments (line 26)
    
    # Obtaining an instance of the builtin type 'tuple' (line 26)
    tuple_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 26)
    # Adding element type (line 26)
    int_32 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 19), tuple_31, int_32)
    # Adding element type (line 26)
    int_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 19), tuple_31, int_33)
    
    # Processing the call keyword arguments (line 26)
    kwargs_34 = {}
    # Getting the type of 'iter' (line 26)
    iter_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 13), 'iter', False)
    # Calling iter(args, kwargs) (line 26)
    iter_call_result_35 = invoke(stypy.reporting.localization.Localization(__file__, 26, 13), iter_30, *[tuple_31], **kwargs_34)
    
    # Assigning a type to the variable 'tuple_' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'tuple_', iter_call_result_35)
    
    # Assigning a Call to a Name (line 27):
    
    # Call to iter(...): (line 27)
    # Processing the call arguments (line 27)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 27)
    # Processing the call arguments (line 27)
    int_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 36), 'int')
    # Processing the call keyword arguments (line 27)
    kwargs_42 = {}
    # Getting the type of 'range' (line 27)
    range_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 30), 'range', False)
    # Calling range(args, kwargs) (line 27)
    range_call_result_43 = invoke(stypy.reporting.localization.Localization(__file__, 27, 30), range_40, *[int_41], **kwargs_42)
    
    comprehension_44 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 14), range_call_result_43)
    # Assigning a type to the variable 'x' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 14), 'x', comprehension_44)
    # Getting the type of 'x' (line 27)
    x_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 14), 'x', False)
    int_38 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 19), 'int')
    # Applying the binary operator '**' (line 27)
    result_pow_39 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 14), '**', x_37, int_38)
    
    list_45 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 14), list_45, result_pow_39)
    # Processing the call keyword arguments (line 27)
    kwargs_46 = {}
    # Getting the type of 'iter' (line 27)
    iter_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'iter', False)
    # Calling iter(args, kwargs) (line 27)
    iter_call_result_47 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), iter_36, *[list_45], **kwargs_46)
    
    # Assigning a type to the variable 'S' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'S', iter_call_result_47)
    
    # Assigning a Call to a Name (line 28):
    
    # Call to iter(...): (line 28)
    # Processing the call arguments (line 28)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 28)
    # Processing the call arguments (line 28)
    int_54 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 36), 'int')
    # Processing the call keyword arguments (line 28)
    kwargs_55 = {}
    # Getting the type of 'range' (line 28)
    range_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 30), 'range', False)
    # Calling range(args, kwargs) (line 28)
    range_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 28, 30), range_53, *[int_54], **kwargs_55)
    
    comprehension_57 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 14), range_call_result_56)
    # Assigning a type to the variable 'i' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 14), 'i', comprehension_57)
    
    # Call to str(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'i' (line 28)
    i_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), 'i', False)
    # Processing the call keyword arguments (line 28)
    kwargs_51 = {}
    # Getting the type of 'str' (line 28)
    str_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 14), 'str', False)
    # Calling str(args, kwargs) (line 28)
    str_call_result_52 = invoke(stypy.reporting.localization.Localization(__file__, 28, 14), str_49, *[i_50], **kwargs_51)
    
    list_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 14), list_58, str_call_result_52)
    # Processing the call keyword arguments (line 28)
    kwargs_59 = {}
    # Getting the type of 'iter' (line 28)
    iter_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'iter', False)
    # Calling iter(args, kwargs) (line 28)
    iter_call_result_60 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), iter_48, *[list_58], **kwargs_59)
    
    # Assigning a type to the variable 'V' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'V', iter_call_result_60)
    
    # Assigning a Call to a Name (line 29):
    
    # Call to iter(...): (line 29)
    # Processing the call arguments (line 29)
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_62 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    int_63 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 23), list_62, int_63)
    # Adding element type (line 29)
    int_64 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 23), list_62, int_64)
    # Adding element type (line 29)
    int_65 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 23), list_62, int_65)
    
    # Processing the call keyword arguments (line 29)
    kwargs_66 = {}
    # Getting the type of 'iter' (line 29)
    iter_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 18), 'iter', False)
    # Calling iter(args, kwargs) (line 29)
    iter_call_result_67 = invoke(stypy.reporting.localization.Localization(__file__, 29, 18), iter_61, *[list_62], **kwargs_66)
    
    # Assigning a type to the variable 'normal_list' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'normal_list', iter_call_result_67)
    
    # Assigning a Call to a Name (line 31):
    
    # Call to fun1(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'S' (line 31)
    S_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'S', False)
    # Processing the call keyword arguments (line 31)
    kwargs_70 = {}
    # Getting the type of 'fun1' (line 31)
    fun1_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 9), 'fun1', False)
    # Calling fun1(args, kwargs) (line 31)
    fun1_call_result_71 = invoke(stypy.reporting.localization.Localization(__file__, 31, 9), fun1_68, *[S_69], **kwargs_70)
    
    # Assigning a type to the variable 'r2' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'r2', fun1_call_result_71)
    
    # Assigning a Call to a Name (line 32):
    
    # Call to fun1b(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'normal_list' (line 32)
    normal_list_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'normal_list', False)
    # Processing the call keyword arguments (line 32)
    kwargs_74 = {}
    # Getting the type of 'fun1b' (line 32)
    fun1b_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 9), 'fun1b', False)
    # Calling fun1b(args, kwargs) (line 32)
    fun1b_call_result_75 = invoke(stypy.reporting.localization.Localization(__file__, 32, 9), fun1b_72, *[normal_list_73], **kwargs_74)
    
    # Assigning a type to the variable 'r3' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'r3', fun1b_call_result_75)
    
    # Assigning a Call to a Name (line 33):
    
    # Call to fun1c(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'tuple_' (line 33)
    tuple__77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'tuple_', False)
    # Processing the call keyword arguments (line 33)
    kwargs_78 = {}
    # Getting the type of 'fun1c' (line 33)
    fun1c_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 9), 'fun1c', False)
    # Calling fun1c(args, kwargs) (line 33)
    fun1c_call_result_79 = invoke(stypy.reporting.localization.Localization(__file__, 33, 9), fun1c_76, *[tuple__77], **kwargs_78)
    
    # Assigning a type to the variable 'r4' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'r4', fun1c_call_result_79)
    
    # Assigning a Call to a Name (line 34):
    
    # Call to fun2(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'V' (line 34)
    V_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 14), 'V', False)
    # Processing the call keyword arguments (line 34)
    kwargs_82 = {}
    # Getting the type of 'fun2' (line 34)
    fun2_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 9), 'fun2', False)
    # Calling fun2(args, kwargs) (line 34)
    fun2_call_result_83 = invoke(stypy.reporting.localization.Localization(__file__, 34, 9), fun2_80, *[V_81], **kwargs_82)
    
    # Assigning a type to the variable 'r5' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'r5', fun2_call_result_83)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
