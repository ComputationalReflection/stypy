
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: theInt = 3
2: theStr = "hi"
3: if True:
4:     union = 3
5: else:
6:     union = "hi"
7: 
8: 
9: def simple_if_hasattr_variant1(a):
10:     b = "hi"
11:     if not '__div__' in a.__class__.__dict__:
12:         r = a / 3
13:         r2 = a[0]
14:         b = 3
15:     r3 = a / 3
16:     r4 = b / 3
17: 
18: 
19: def simple_if_hasattr_variant2(a):
20:     b = "hi"
21:     if not type(a).__dict__.has_key('__div__'):
22:         r = a / 3
23:         r2 = a[0]
24:         b = 3
25:     r3 = a / 3
26:     r4 = b / 3
27: 
28: 
29: def simple_if_hasattr_variant3(a):
30:     b = "hi"
31:     if not '__div__' in type(a).__dict__:
32:         r = a / 3
33:         r2 = a[0]
34:         b = 3
35:     r3 = a / 3
36:     r4 = b / 3
37: 
38: 
39: def simple_if_hasattr_variant4(a):
40:     b = "hi"
41:     if not '__div__' in dir(type(a)):
42:         r = a / 3
43:         r2 = a[0]
44:         b = 3
45:     r3 = a / 3
46:     r4 = b / 3
47: 
48: 
49: simple_if_hasattr_variant1(union)
50: simple_if_hasattr_variant2(union)
51: simple_if_hasattr_variant3(union)
52: simple_if_hasattr_variant4(union)

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 1):
int_3867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 9), 'int')
# Assigning a type to the variable 'theInt' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'theInt', int_3867)

# Assigning a Str to a Name (line 2):
str_3868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 9), 'str', 'hi')
# Assigning a type to the variable 'theStr' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'theStr', str_3868)

# Getting the type of 'True' (line 3)
True_3869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 3), 'True')
# Testing the type of an if condition (line 3)
if_condition_3870 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 3, 0), True_3869)
# Assigning a type to the variable 'if_condition_3870' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'if_condition_3870', if_condition_3870)
# SSA begins for if statement (line 3)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Num to a Name (line 4):
int_3871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 12), 'int')
# Assigning a type to the variable 'union' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'union', int_3871)
# SSA branch for the else part of an if statement (line 3)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 6):
str_3872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'str', 'hi')
# Assigning a type to the variable 'union' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'union', str_3872)
# SSA join for if statement (line 3)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def simple_if_hasattr_variant1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_hasattr_variant1'
    module_type_store = module_type_store.open_function_context('simple_if_hasattr_variant1', 9, 0, False)
    
    # Passed parameters checking function
    simple_if_hasattr_variant1.stypy_localization = localization
    simple_if_hasattr_variant1.stypy_type_of_self = None
    simple_if_hasattr_variant1.stypy_type_store = module_type_store
    simple_if_hasattr_variant1.stypy_function_name = 'simple_if_hasattr_variant1'
    simple_if_hasattr_variant1.stypy_param_names_list = ['a']
    simple_if_hasattr_variant1.stypy_varargs_param_name = None
    simple_if_hasattr_variant1.stypy_kwargs_param_name = None
    simple_if_hasattr_variant1.stypy_call_defaults = defaults
    simple_if_hasattr_variant1.stypy_call_varargs = varargs
    simple_if_hasattr_variant1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_hasattr_variant1', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_hasattr_variant1', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_hasattr_variant1(...)' code ##################

    
    # Assigning a Str to a Name (line 10):
    str_3873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'b', str_3873)
    
    # Type idiom detected: calculating its left and rigth part (line 11)
    str_3874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 11), 'str', '__div__')
    # Getting the type of 'a' (line 11)
    a_3875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 24), 'a')
    
    (may_be_3876, more_types_in_union_3877) = may_not_provide_member(str_3874, a_3875)

    if may_be_3876:

        if more_types_in_union_3877:
            # Runtime conditional SSA (line 11)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'a', remove_member_provider_from_union(a_3875, '__div__'))
        
        # Assigning a BinOp to a Name (line 12):
        # Getting the type of 'a' (line 12)
        a_3878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'a')
        int_3879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 16), 'int')
        # Applying the binary operator 'div' (line 12)
        result_div_3880 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 12), 'div', a_3878, int_3879)
        
        # Assigning a type to the variable 'r' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'r', result_div_3880)
        
        # Assigning a Subscript to a Name (line 13):
        
        # Obtaining the type of the subscript
        int_3881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'int')
        # Getting the type of 'a' (line 13)
        a_3882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 13)
        getitem___3883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 13), a_3882, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 13)
        subscript_call_result_3884 = invoke(stypy.reporting.localization.Localization(__file__, 13, 13), getitem___3883, int_3881)
        
        # Assigning a type to the variable 'r2' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'r2', subscript_call_result_3884)
        
        # Assigning a Num to a Name (line 14):
        int_3885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'int')
        # Assigning a type to the variable 'b' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'b', int_3885)

        if more_types_in_union_3877:
            # SSA join for if statement (line 11)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 15):
    # Getting the type of 'a' (line 15)
    a_3886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 9), 'a')
    int_3887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 13), 'int')
    # Applying the binary operator 'div' (line 15)
    result_div_3888 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 9), 'div', a_3886, int_3887)
    
    # Assigning a type to the variable 'r3' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'r3', result_div_3888)
    
    # Assigning a BinOp to a Name (line 16):
    # Getting the type of 'b' (line 16)
    b_3889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 9), 'b')
    int_3890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'int')
    # Applying the binary operator 'div' (line 16)
    result_div_3891 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 9), 'div', b_3889, int_3890)
    
    # Assigning a type to the variable 'r4' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'r4', result_div_3891)
    
    # ################# End of 'simple_if_hasattr_variant1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_hasattr_variant1' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_3892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3892)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_hasattr_variant1'
    return stypy_return_type_3892

# Assigning a type to the variable 'simple_if_hasattr_variant1' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'simple_if_hasattr_variant1', simple_if_hasattr_variant1)

@norecursion
def simple_if_hasattr_variant2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_hasattr_variant2'
    module_type_store = module_type_store.open_function_context('simple_if_hasattr_variant2', 19, 0, False)
    
    # Passed parameters checking function
    simple_if_hasattr_variant2.stypy_localization = localization
    simple_if_hasattr_variant2.stypy_type_of_self = None
    simple_if_hasattr_variant2.stypy_type_store = module_type_store
    simple_if_hasattr_variant2.stypy_function_name = 'simple_if_hasattr_variant2'
    simple_if_hasattr_variant2.stypy_param_names_list = ['a']
    simple_if_hasattr_variant2.stypy_varargs_param_name = None
    simple_if_hasattr_variant2.stypy_kwargs_param_name = None
    simple_if_hasattr_variant2.stypy_call_defaults = defaults
    simple_if_hasattr_variant2.stypy_call_varargs = varargs
    simple_if_hasattr_variant2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_hasattr_variant2', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_hasattr_variant2', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_hasattr_variant2(...)' code ##################

    
    # Assigning a Str to a Name (line 20):
    str_3893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'b', str_3893)
    
    # Type idiom detected: calculating its left and rigth part (line 21)
    str_3894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 36), 'str', '__div__')
    # Getting the type of 'a' (line 21)
    a_3895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'a')
    
    (may_be_3896, more_types_in_union_3897) = may_not_provide_member(str_3894, a_3895)

    if may_be_3896:

        if more_types_in_union_3897:
            # Runtime conditional SSA (line 21)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'a', remove_member_provider_from_union(a_3895, '__div__'))
        
        # Assigning a BinOp to a Name (line 22):
        # Getting the type of 'a' (line 22)
        a_3898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'a')
        int_3899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'int')
        # Applying the binary operator 'div' (line 22)
        result_div_3900 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 12), 'div', a_3898, int_3899)
        
        # Assigning a type to the variable 'r' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'r', result_div_3900)
        
        # Assigning a Subscript to a Name (line 23):
        
        # Obtaining the type of the subscript
        int_3901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'int')
        # Getting the type of 'a' (line 23)
        a_3902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 23)
        getitem___3903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 13), a_3902, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 23)
        subscript_call_result_3904 = invoke(stypy.reporting.localization.Localization(__file__, 23, 13), getitem___3903, int_3901)
        
        # Assigning a type to the variable 'r2' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'r2', subscript_call_result_3904)
        
        # Assigning a Num to a Name (line 24):
        int_3905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 12), 'int')
        # Assigning a type to the variable 'b' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'b', int_3905)

        if more_types_in_union_3897:
            # SSA join for if statement (line 21)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 25):
    # Getting the type of 'a' (line 25)
    a_3906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 9), 'a')
    int_3907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'int')
    # Applying the binary operator 'div' (line 25)
    result_div_3908 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 9), 'div', a_3906, int_3907)
    
    # Assigning a type to the variable 'r3' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'r3', result_div_3908)
    
    # Assigning a BinOp to a Name (line 26):
    # Getting the type of 'b' (line 26)
    b_3909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 9), 'b')
    int_3910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 13), 'int')
    # Applying the binary operator 'div' (line 26)
    result_div_3911 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 9), 'div', b_3909, int_3910)
    
    # Assigning a type to the variable 'r4' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'r4', result_div_3911)
    
    # ################# End of 'simple_if_hasattr_variant2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_hasattr_variant2' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_3912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3912)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_hasattr_variant2'
    return stypy_return_type_3912

# Assigning a type to the variable 'simple_if_hasattr_variant2' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'simple_if_hasattr_variant2', simple_if_hasattr_variant2)

@norecursion
def simple_if_hasattr_variant3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_hasattr_variant3'
    module_type_store = module_type_store.open_function_context('simple_if_hasattr_variant3', 29, 0, False)
    
    # Passed parameters checking function
    simple_if_hasattr_variant3.stypy_localization = localization
    simple_if_hasattr_variant3.stypy_type_of_self = None
    simple_if_hasattr_variant3.stypy_type_store = module_type_store
    simple_if_hasattr_variant3.stypy_function_name = 'simple_if_hasattr_variant3'
    simple_if_hasattr_variant3.stypy_param_names_list = ['a']
    simple_if_hasattr_variant3.stypy_varargs_param_name = None
    simple_if_hasattr_variant3.stypy_kwargs_param_name = None
    simple_if_hasattr_variant3.stypy_call_defaults = defaults
    simple_if_hasattr_variant3.stypy_call_varargs = varargs
    simple_if_hasattr_variant3.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_hasattr_variant3', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_hasattr_variant3', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_hasattr_variant3(...)' code ##################

    
    # Assigning a Str to a Name (line 30):
    str_3913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'b', str_3913)
    
    # Type idiom detected: calculating its left and rigth part (line 31)
    str_3914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 11), 'str', '__div__')
    # Getting the type of 'a' (line 31)
    a_3915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 29), 'a')
    
    (may_be_3916, more_types_in_union_3917) = may_not_provide_member(str_3914, a_3915)

    if may_be_3916:

        if more_types_in_union_3917:
            # Runtime conditional SSA (line 31)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'a', remove_member_provider_from_union(a_3915, '__div__'))
        
        # Assigning a BinOp to a Name (line 32):
        # Getting the type of 'a' (line 32)
        a_3918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'a')
        int_3919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 16), 'int')
        # Applying the binary operator 'div' (line 32)
        result_div_3920 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 12), 'div', a_3918, int_3919)
        
        # Assigning a type to the variable 'r' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'r', result_div_3920)
        
        # Assigning a Subscript to a Name (line 33):
        
        # Obtaining the type of the subscript
        int_3921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 15), 'int')
        # Getting the type of 'a' (line 33)
        a_3922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 33)
        getitem___3923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 13), a_3922, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 33)
        subscript_call_result_3924 = invoke(stypy.reporting.localization.Localization(__file__, 33, 13), getitem___3923, int_3921)
        
        # Assigning a type to the variable 'r2' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'r2', subscript_call_result_3924)
        
        # Assigning a Num to a Name (line 34):
        int_3925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 12), 'int')
        # Assigning a type to the variable 'b' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'b', int_3925)

        if more_types_in_union_3917:
            # SSA join for if statement (line 31)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 35):
    # Getting the type of 'a' (line 35)
    a_3926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 9), 'a')
    int_3927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 13), 'int')
    # Applying the binary operator 'div' (line 35)
    result_div_3928 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 9), 'div', a_3926, int_3927)
    
    # Assigning a type to the variable 'r3' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'r3', result_div_3928)
    
    # Assigning a BinOp to a Name (line 36):
    # Getting the type of 'b' (line 36)
    b_3929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 9), 'b')
    int_3930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 13), 'int')
    # Applying the binary operator 'div' (line 36)
    result_div_3931 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 9), 'div', b_3929, int_3930)
    
    # Assigning a type to the variable 'r4' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'r4', result_div_3931)
    
    # ################# End of 'simple_if_hasattr_variant3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_hasattr_variant3' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_3932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3932)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_hasattr_variant3'
    return stypy_return_type_3932

# Assigning a type to the variable 'simple_if_hasattr_variant3' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'simple_if_hasattr_variant3', simple_if_hasattr_variant3)

@norecursion
def simple_if_hasattr_variant4(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_hasattr_variant4'
    module_type_store = module_type_store.open_function_context('simple_if_hasattr_variant4', 39, 0, False)
    
    # Passed parameters checking function
    simple_if_hasattr_variant4.stypy_localization = localization
    simple_if_hasattr_variant4.stypy_type_of_self = None
    simple_if_hasattr_variant4.stypy_type_store = module_type_store
    simple_if_hasattr_variant4.stypy_function_name = 'simple_if_hasattr_variant4'
    simple_if_hasattr_variant4.stypy_param_names_list = ['a']
    simple_if_hasattr_variant4.stypy_varargs_param_name = None
    simple_if_hasattr_variant4.stypy_kwargs_param_name = None
    simple_if_hasattr_variant4.stypy_call_defaults = defaults
    simple_if_hasattr_variant4.stypy_call_varargs = varargs
    simple_if_hasattr_variant4.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_hasattr_variant4', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_hasattr_variant4', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_hasattr_variant4(...)' code ##################

    
    # Assigning a Str to a Name (line 40):
    str_3933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'b', str_3933)
    
    # Type idiom detected: calculating its left and rigth part (line 41)
    str_3934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 11), 'str', '__div__')
    # Getting the type of 'a' (line 41)
    a_3935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 33), 'a')
    
    (may_be_3936, more_types_in_union_3937) = may_not_provide_member(str_3934, a_3935)

    if may_be_3936:

        if more_types_in_union_3937:
            # Runtime conditional SSA (line 41)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'a', remove_member_provider_from_union(a_3935, '__div__'))
        
        # Assigning a BinOp to a Name (line 42):
        # Getting the type of 'a' (line 42)
        a_3938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'a')
        int_3939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 16), 'int')
        # Applying the binary operator 'div' (line 42)
        result_div_3940 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 12), 'div', a_3938, int_3939)
        
        # Assigning a type to the variable 'r' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'r', result_div_3940)
        
        # Assigning a Subscript to a Name (line 43):
        
        # Obtaining the type of the subscript
        int_3941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 15), 'int')
        # Getting the type of 'a' (line 43)
        a_3942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 43)
        getitem___3943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 13), a_3942, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 43)
        subscript_call_result_3944 = invoke(stypy.reporting.localization.Localization(__file__, 43, 13), getitem___3943, int_3941)
        
        # Assigning a type to the variable 'r2' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'r2', subscript_call_result_3944)
        
        # Assigning a Num to a Name (line 44):
        int_3945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 12), 'int')
        # Assigning a type to the variable 'b' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'b', int_3945)

        if more_types_in_union_3937:
            # SSA join for if statement (line 41)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 45):
    # Getting the type of 'a' (line 45)
    a_3946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 9), 'a')
    int_3947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 13), 'int')
    # Applying the binary operator 'div' (line 45)
    result_div_3948 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 9), 'div', a_3946, int_3947)
    
    # Assigning a type to the variable 'r3' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'r3', result_div_3948)
    
    # Assigning a BinOp to a Name (line 46):
    # Getting the type of 'b' (line 46)
    b_3949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 9), 'b')
    int_3950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 13), 'int')
    # Applying the binary operator 'div' (line 46)
    result_div_3951 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 9), 'div', b_3949, int_3950)
    
    # Assigning a type to the variable 'r4' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'r4', result_div_3951)
    
    # ################# End of 'simple_if_hasattr_variant4(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_hasattr_variant4' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_3952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3952)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_hasattr_variant4'
    return stypy_return_type_3952

# Assigning a type to the variable 'simple_if_hasattr_variant4' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'simple_if_hasattr_variant4', simple_if_hasattr_variant4)

# Call to simple_if_hasattr_variant1(...): (line 49)
# Processing the call arguments (line 49)
# Getting the type of 'union' (line 49)
union_3954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 27), 'union', False)
# Processing the call keyword arguments (line 49)
kwargs_3955 = {}
# Getting the type of 'simple_if_hasattr_variant1' (line 49)
simple_if_hasattr_variant1_3953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'simple_if_hasattr_variant1', False)
# Calling simple_if_hasattr_variant1(args, kwargs) (line 49)
simple_if_hasattr_variant1_call_result_3956 = invoke(stypy.reporting.localization.Localization(__file__, 49, 0), simple_if_hasattr_variant1_3953, *[union_3954], **kwargs_3955)


# Call to simple_if_hasattr_variant2(...): (line 50)
# Processing the call arguments (line 50)
# Getting the type of 'union' (line 50)
union_3958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 27), 'union', False)
# Processing the call keyword arguments (line 50)
kwargs_3959 = {}
# Getting the type of 'simple_if_hasattr_variant2' (line 50)
simple_if_hasattr_variant2_3957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'simple_if_hasattr_variant2', False)
# Calling simple_if_hasattr_variant2(args, kwargs) (line 50)
simple_if_hasattr_variant2_call_result_3960 = invoke(stypy.reporting.localization.Localization(__file__, 50, 0), simple_if_hasattr_variant2_3957, *[union_3958], **kwargs_3959)


# Call to simple_if_hasattr_variant3(...): (line 51)
# Processing the call arguments (line 51)
# Getting the type of 'union' (line 51)
union_3962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 27), 'union', False)
# Processing the call keyword arguments (line 51)
kwargs_3963 = {}
# Getting the type of 'simple_if_hasattr_variant3' (line 51)
simple_if_hasattr_variant3_3961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'simple_if_hasattr_variant3', False)
# Calling simple_if_hasattr_variant3(args, kwargs) (line 51)
simple_if_hasattr_variant3_call_result_3964 = invoke(stypy.reporting.localization.Localization(__file__, 51, 0), simple_if_hasattr_variant3_3961, *[union_3962], **kwargs_3963)


# Call to simple_if_hasattr_variant4(...): (line 52)
# Processing the call arguments (line 52)
# Getting the type of 'union' (line 52)
union_3966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 27), 'union', False)
# Processing the call keyword arguments (line 52)
kwargs_3967 = {}
# Getting the type of 'simple_if_hasattr_variant4' (line 52)
simple_if_hasattr_variant4_3965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'simple_if_hasattr_variant4', False)
# Calling simple_if_hasattr_variant4(args, kwargs) (line 52)
simple_if_hasattr_variant4_call_result_3968 = invoke(stypy.reporting.localization.Localization(__file__, 52, 0), simple_if_hasattr_variant4_3965, *[union_3966], **kwargs_3967)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
