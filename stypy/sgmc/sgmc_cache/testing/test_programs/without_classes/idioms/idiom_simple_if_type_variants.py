
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
9: def simple_if_variant1(a):
10:     b = "hi"
11: 
12:     if type(a) == int:
13:         r = a / 3
14:         r2 = a[0]
15:         b = 3
16:     r3 = a / 3
17:     r4 = b / 3
18: 
19: 
20: def simple_if_variant2(a):
21:     b = "hi"
22: 
23:     if a.__class__ is int:
24:         r = a / 3
25:         r2 = a[0]
26:         b = 3
27:     r3 = a / 3
28:     r4 = b / 3
29: 
30: 
31: def simple_if_variant3(a):
32:     b = "hi"
33:     if a.__class__ == int:
34:         r = a / 3
35:         r2 = a[0]
36:         b = 3
37:     r3 = a / 3
38:     r4 = b / 3
39: 
40: 
41: def simple_if_variant4(a):
42:     b = "hi"
43:     if int is type(a):
44:         r = a / 3
45:         r2 = a[0]
46:         b = 3
47:     r3 = a / 3
48:     r4 = b / 3
49: 
50: 
51: def simple_if_variant5(a):
52:     b = "hi"
53:     if int == type(a):
54:         r = a / 3
55:         r2 = a[0]
56:         b = 3
57:     r3 = a / 3
58:     r4 = b / 3
59: 
60: 
61: def simple_if_variant6(a):
62:     b = "hi"
63:     if int is a.__class__:
64:         r = a / 3
65:         r2 = a[0]
66:         b = 3
67:     r3 = a / 3
68:     r4 = b / 3
69: 
70: 
71: def simple_if_variant7(a):
72:     b = "hi"
73:     if int == a.__class__:
74:         r = a / 3
75:         r2 = a[0]
76:         b = 3
77:     r3 = a / 3
78:     r4 = b / 3
79: 
80: 
81: simple_if_variant1(union)
82: simple_if_variant2(union)
83: simple_if_variant3(union)
84: simple_if_variant4(union)
85: simple_if_variant5(union)
86: simple_if_variant6(union)
87: simple_if_variant7(union)
88: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 1):
int_4714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 9), 'int')
# Assigning a type to the variable 'theInt' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'theInt', int_4714)

# Assigning a Str to a Name (line 2):
str_4715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 9), 'str', 'hi')
# Assigning a type to the variable 'theStr' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'theStr', str_4715)

# Getting the type of 'True' (line 3)
True_4716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 3), 'True')
# Testing the type of an if condition (line 3)
if_condition_4717 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 3, 0), True_4716)
# Assigning a type to the variable 'if_condition_4717' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'if_condition_4717', if_condition_4717)
# SSA begins for if statement (line 3)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Num to a Name (line 4):
int_4718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 12), 'int')
# Assigning a type to the variable 'union' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'union', int_4718)
# SSA branch for the else part of an if statement (line 3)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 6):
str_4719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'str', 'hi')
# Assigning a type to the variable 'union' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'union', str_4719)
# SSA join for if statement (line 3)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def simple_if_variant1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_variant1'
    module_type_store = module_type_store.open_function_context('simple_if_variant1', 9, 0, False)
    
    # Passed parameters checking function
    simple_if_variant1.stypy_localization = localization
    simple_if_variant1.stypy_type_of_self = None
    simple_if_variant1.stypy_type_store = module_type_store
    simple_if_variant1.stypy_function_name = 'simple_if_variant1'
    simple_if_variant1.stypy_param_names_list = ['a']
    simple_if_variant1.stypy_varargs_param_name = None
    simple_if_variant1.stypy_kwargs_param_name = None
    simple_if_variant1.stypy_call_defaults = defaults
    simple_if_variant1.stypy_call_varargs = varargs
    simple_if_variant1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_variant1', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_variant1', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_variant1(...)' code ##################

    
    # Assigning a Str to a Name (line 10):
    str_4720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'b', str_4720)
    
    # Type idiom detected: calculating its left and rigth part (line 12)
    # Getting the type of 'a' (line 12)
    a_4721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'a')
    # Getting the type of 'int' (line 12)
    int_4722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 18), 'int')
    
    (may_be_4723, more_types_in_union_4724) = may_be_type(a_4721, int_4722)

    if may_be_4723:

        if more_types_in_union_4724:
            # Runtime conditional SSA (line 12)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'a', int_4722())
        
        # Assigning a BinOp to a Name (line 13):
        # Getting the type of 'a' (line 13)
        a_4725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'a')
        int_4726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 16), 'int')
        # Applying the binary operator 'div' (line 13)
        result_div_4727 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 12), 'div', a_4725, int_4726)
        
        # Assigning a type to the variable 'r' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'r', result_div_4727)
        
        # Assigning a Subscript to a Name (line 14):
        
        # Obtaining the type of the subscript
        int_4728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'int')
        # Getting the type of 'a' (line 14)
        a_4729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 14)
        getitem___4730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 13), a_4729, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 14)
        subscript_call_result_4731 = invoke(stypy.reporting.localization.Localization(__file__, 14, 13), getitem___4730, int_4728)
        
        # Assigning a type to the variable 'r2' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'r2', subscript_call_result_4731)
        
        # Assigning a Num to a Name (line 15):
        int_4732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 12), 'int')
        # Assigning a type to the variable 'b' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'b', int_4732)

        if more_types_in_union_4724:
            # SSA join for if statement (line 12)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 16):
    # Getting the type of 'a' (line 16)
    a_4733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 9), 'a')
    int_4734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'int')
    # Applying the binary operator 'div' (line 16)
    result_div_4735 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 9), 'div', a_4733, int_4734)
    
    # Assigning a type to the variable 'r3' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'r3', result_div_4735)
    
    # Assigning a BinOp to a Name (line 17):
    # Getting the type of 'b' (line 17)
    b_4736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 9), 'b')
    int_4737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 13), 'int')
    # Applying the binary operator 'div' (line 17)
    result_div_4738 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 9), 'div', b_4736, int_4737)
    
    # Assigning a type to the variable 'r4' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'r4', result_div_4738)
    
    # ################# End of 'simple_if_variant1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_variant1' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_4739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4739)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_variant1'
    return stypy_return_type_4739

# Assigning a type to the variable 'simple_if_variant1' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'simple_if_variant1', simple_if_variant1)

@norecursion
def simple_if_variant2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_variant2'
    module_type_store = module_type_store.open_function_context('simple_if_variant2', 20, 0, False)
    
    # Passed parameters checking function
    simple_if_variant2.stypy_localization = localization
    simple_if_variant2.stypy_type_of_self = None
    simple_if_variant2.stypy_type_store = module_type_store
    simple_if_variant2.stypy_function_name = 'simple_if_variant2'
    simple_if_variant2.stypy_param_names_list = ['a']
    simple_if_variant2.stypy_varargs_param_name = None
    simple_if_variant2.stypy_kwargs_param_name = None
    simple_if_variant2.stypy_call_defaults = defaults
    simple_if_variant2.stypy_call_varargs = varargs
    simple_if_variant2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_variant2', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_variant2', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_variant2(...)' code ##################

    
    # Assigning a Str to a Name (line 21):
    str_4740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'b', str_4740)
    
    # Type idiom detected: calculating its left and rigth part (line 23)
    # Getting the type of 'a' (line 23)
    a_4741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 7), 'a')
    # Getting the type of 'int' (line 23)
    int_4742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 22), 'int')
    
    (may_be_4743, more_types_in_union_4744) = may_be_type(a_4741, int_4742)

    if may_be_4743:

        if more_types_in_union_4744:
            # Runtime conditional SSA (line 23)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'a', int_4742())
        
        # Assigning a BinOp to a Name (line 24):
        # Getting the type of 'a' (line 24)
        a_4745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'a')
        int_4746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 16), 'int')
        # Applying the binary operator 'div' (line 24)
        result_div_4747 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 12), 'div', a_4745, int_4746)
        
        # Assigning a type to the variable 'r' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'r', result_div_4747)
        
        # Assigning a Subscript to a Name (line 25):
        
        # Obtaining the type of the subscript
        int_4748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 15), 'int')
        # Getting the type of 'a' (line 25)
        a_4749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 25)
        getitem___4750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 13), a_4749, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 25)
        subscript_call_result_4751 = invoke(stypy.reporting.localization.Localization(__file__, 25, 13), getitem___4750, int_4748)
        
        # Assigning a type to the variable 'r2' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'r2', subscript_call_result_4751)
        
        # Assigning a Num to a Name (line 26):
        int_4752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 12), 'int')
        # Assigning a type to the variable 'b' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'b', int_4752)

        if more_types_in_union_4744:
            # SSA join for if statement (line 23)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 27):
    # Getting the type of 'a' (line 27)
    a_4753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 9), 'a')
    int_4754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 13), 'int')
    # Applying the binary operator 'div' (line 27)
    result_div_4755 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 9), 'div', a_4753, int_4754)
    
    # Assigning a type to the variable 'r3' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'r3', result_div_4755)
    
    # Assigning a BinOp to a Name (line 28):
    # Getting the type of 'b' (line 28)
    b_4756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 9), 'b')
    int_4757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 13), 'int')
    # Applying the binary operator 'div' (line 28)
    result_div_4758 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 9), 'div', b_4756, int_4757)
    
    # Assigning a type to the variable 'r4' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'r4', result_div_4758)
    
    # ################# End of 'simple_if_variant2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_variant2' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_4759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4759)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_variant2'
    return stypy_return_type_4759

# Assigning a type to the variable 'simple_if_variant2' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'simple_if_variant2', simple_if_variant2)

@norecursion
def simple_if_variant3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_variant3'
    module_type_store = module_type_store.open_function_context('simple_if_variant3', 31, 0, False)
    
    # Passed parameters checking function
    simple_if_variant3.stypy_localization = localization
    simple_if_variant3.stypy_type_of_self = None
    simple_if_variant3.stypy_type_store = module_type_store
    simple_if_variant3.stypy_function_name = 'simple_if_variant3'
    simple_if_variant3.stypy_param_names_list = ['a']
    simple_if_variant3.stypy_varargs_param_name = None
    simple_if_variant3.stypy_kwargs_param_name = None
    simple_if_variant3.stypy_call_defaults = defaults
    simple_if_variant3.stypy_call_varargs = varargs
    simple_if_variant3.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_variant3', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_variant3', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_variant3(...)' code ##################

    
    # Assigning a Str to a Name (line 32):
    str_4760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'b', str_4760)
    
    # Type idiom detected: calculating its left and rigth part (line 33)
    # Getting the type of 'a' (line 33)
    a_4761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 7), 'a')
    # Getting the type of 'int' (line 33)
    int_4762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 22), 'int')
    
    (may_be_4763, more_types_in_union_4764) = may_be_type(a_4761, int_4762)

    if may_be_4763:

        if more_types_in_union_4764:
            # Runtime conditional SSA (line 33)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'a', int_4762())
        
        # Assigning a BinOp to a Name (line 34):
        # Getting the type of 'a' (line 34)
        a_4765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'a')
        int_4766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 16), 'int')
        # Applying the binary operator 'div' (line 34)
        result_div_4767 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 12), 'div', a_4765, int_4766)
        
        # Assigning a type to the variable 'r' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'r', result_div_4767)
        
        # Assigning a Subscript to a Name (line 35):
        
        # Obtaining the type of the subscript
        int_4768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 15), 'int')
        # Getting the type of 'a' (line 35)
        a_4769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 35)
        getitem___4770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 13), a_4769, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 35)
        subscript_call_result_4771 = invoke(stypy.reporting.localization.Localization(__file__, 35, 13), getitem___4770, int_4768)
        
        # Assigning a type to the variable 'r2' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'r2', subscript_call_result_4771)
        
        # Assigning a Num to a Name (line 36):
        int_4772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 12), 'int')
        # Assigning a type to the variable 'b' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'b', int_4772)

        if more_types_in_union_4764:
            # SSA join for if statement (line 33)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 37):
    # Getting the type of 'a' (line 37)
    a_4773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 9), 'a')
    int_4774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 13), 'int')
    # Applying the binary operator 'div' (line 37)
    result_div_4775 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 9), 'div', a_4773, int_4774)
    
    # Assigning a type to the variable 'r3' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'r3', result_div_4775)
    
    # Assigning a BinOp to a Name (line 38):
    # Getting the type of 'b' (line 38)
    b_4776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 9), 'b')
    int_4777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 13), 'int')
    # Applying the binary operator 'div' (line 38)
    result_div_4778 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 9), 'div', b_4776, int_4777)
    
    # Assigning a type to the variable 'r4' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'r4', result_div_4778)
    
    # ################# End of 'simple_if_variant3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_variant3' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_4779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4779)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_variant3'
    return stypy_return_type_4779

# Assigning a type to the variable 'simple_if_variant3' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'simple_if_variant3', simple_if_variant3)

@norecursion
def simple_if_variant4(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_variant4'
    module_type_store = module_type_store.open_function_context('simple_if_variant4', 41, 0, False)
    
    # Passed parameters checking function
    simple_if_variant4.stypy_localization = localization
    simple_if_variant4.stypy_type_of_self = None
    simple_if_variant4.stypy_type_store = module_type_store
    simple_if_variant4.stypy_function_name = 'simple_if_variant4'
    simple_if_variant4.stypy_param_names_list = ['a']
    simple_if_variant4.stypy_varargs_param_name = None
    simple_if_variant4.stypy_kwargs_param_name = None
    simple_if_variant4.stypy_call_defaults = defaults
    simple_if_variant4.stypy_call_varargs = varargs
    simple_if_variant4.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_variant4', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_variant4', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_variant4(...)' code ##################

    
    # Assigning a Str to a Name (line 42):
    str_4780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'b', str_4780)
    
    # Type idiom detected: calculating its left and rigth part (line 43)
    # Getting the type of 'a' (line 43)
    a_4781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 19), 'a')
    # Getting the type of 'int' (line 43)
    int_4782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 7), 'int')
    
    (may_be_4783, more_types_in_union_4784) = may_be_type(a_4781, int_4782)

    if may_be_4783:

        if more_types_in_union_4784:
            # Runtime conditional SSA (line 43)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'a', int_4782())
        
        # Assigning a BinOp to a Name (line 44):
        # Getting the type of 'a' (line 44)
        a_4785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'a')
        int_4786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 16), 'int')
        # Applying the binary operator 'div' (line 44)
        result_div_4787 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 12), 'div', a_4785, int_4786)
        
        # Assigning a type to the variable 'r' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'r', result_div_4787)
        
        # Assigning a Subscript to a Name (line 45):
        
        # Obtaining the type of the subscript
        int_4788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 15), 'int')
        # Getting the type of 'a' (line 45)
        a_4789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 45)
        getitem___4790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 13), a_4789, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 45)
        subscript_call_result_4791 = invoke(stypy.reporting.localization.Localization(__file__, 45, 13), getitem___4790, int_4788)
        
        # Assigning a type to the variable 'r2' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'r2', subscript_call_result_4791)
        
        # Assigning a Num to a Name (line 46):
        int_4792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 12), 'int')
        # Assigning a type to the variable 'b' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'b', int_4792)

        if more_types_in_union_4784:
            # SSA join for if statement (line 43)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 47):
    # Getting the type of 'a' (line 47)
    a_4793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 9), 'a')
    int_4794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 13), 'int')
    # Applying the binary operator 'div' (line 47)
    result_div_4795 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 9), 'div', a_4793, int_4794)
    
    # Assigning a type to the variable 'r3' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'r3', result_div_4795)
    
    # Assigning a BinOp to a Name (line 48):
    # Getting the type of 'b' (line 48)
    b_4796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 9), 'b')
    int_4797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 13), 'int')
    # Applying the binary operator 'div' (line 48)
    result_div_4798 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 9), 'div', b_4796, int_4797)
    
    # Assigning a type to the variable 'r4' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'r4', result_div_4798)
    
    # ################# End of 'simple_if_variant4(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_variant4' in the type store
    # Getting the type of 'stypy_return_type' (line 41)
    stypy_return_type_4799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4799)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_variant4'
    return stypy_return_type_4799

# Assigning a type to the variable 'simple_if_variant4' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'simple_if_variant4', simple_if_variant4)

@norecursion
def simple_if_variant5(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_variant5'
    module_type_store = module_type_store.open_function_context('simple_if_variant5', 51, 0, False)
    
    # Passed parameters checking function
    simple_if_variant5.stypy_localization = localization
    simple_if_variant5.stypy_type_of_self = None
    simple_if_variant5.stypy_type_store = module_type_store
    simple_if_variant5.stypy_function_name = 'simple_if_variant5'
    simple_if_variant5.stypy_param_names_list = ['a']
    simple_if_variant5.stypy_varargs_param_name = None
    simple_if_variant5.stypy_kwargs_param_name = None
    simple_if_variant5.stypy_call_defaults = defaults
    simple_if_variant5.stypy_call_varargs = varargs
    simple_if_variant5.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_variant5', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_variant5', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_variant5(...)' code ##################

    
    # Assigning a Str to a Name (line 52):
    str_4800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'b', str_4800)
    
    # Type idiom detected: calculating its left and rigth part (line 53)
    # Getting the type of 'a' (line 53)
    a_4801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 19), 'a')
    # Getting the type of 'int' (line 53)
    int_4802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 7), 'int')
    
    (may_be_4803, more_types_in_union_4804) = may_be_type(a_4801, int_4802)

    if may_be_4803:

        if more_types_in_union_4804:
            # Runtime conditional SSA (line 53)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'a', int_4802())
        
        # Assigning a BinOp to a Name (line 54):
        # Getting the type of 'a' (line 54)
        a_4805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'a')
        int_4806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 16), 'int')
        # Applying the binary operator 'div' (line 54)
        result_div_4807 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 12), 'div', a_4805, int_4806)
        
        # Assigning a type to the variable 'r' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'r', result_div_4807)
        
        # Assigning a Subscript to a Name (line 55):
        
        # Obtaining the type of the subscript
        int_4808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 15), 'int')
        # Getting the type of 'a' (line 55)
        a_4809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 55)
        getitem___4810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 13), a_4809, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 55)
        subscript_call_result_4811 = invoke(stypy.reporting.localization.Localization(__file__, 55, 13), getitem___4810, int_4808)
        
        # Assigning a type to the variable 'r2' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'r2', subscript_call_result_4811)
        
        # Assigning a Num to a Name (line 56):
        int_4812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 12), 'int')
        # Assigning a type to the variable 'b' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'b', int_4812)

        if more_types_in_union_4804:
            # SSA join for if statement (line 53)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 57):
    # Getting the type of 'a' (line 57)
    a_4813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 9), 'a')
    int_4814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 13), 'int')
    # Applying the binary operator 'div' (line 57)
    result_div_4815 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 9), 'div', a_4813, int_4814)
    
    # Assigning a type to the variable 'r3' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'r3', result_div_4815)
    
    # Assigning a BinOp to a Name (line 58):
    # Getting the type of 'b' (line 58)
    b_4816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 9), 'b')
    int_4817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 13), 'int')
    # Applying the binary operator 'div' (line 58)
    result_div_4818 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 9), 'div', b_4816, int_4817)
    
    # Assigning a type to the variable 'r4' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'r4', result_div_4818)
    
    # ################# End of 'simple_if_variant5(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_variant5' in the type store
    # Getting the type of 'stypy_return_type' (line 51)
    stypy_return_type_4819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4819)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_variant5'
    return stypy_return_type_4819

# Assigning a type to the variable 'simple_if_variant5' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'simple_if_variant5', simple_if_variant5)

@norecursion
def simple_if_variant6(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_variant6'
    module_type_store = module_type_store.open_function_context('simple_if_variant6', 61, 0, False)
    
    # Passed parameters checking function
    simple_if_variant6.stypy_localization = localization
    simple_if_variant6.stypy_type_of_self = None
    simple_if_variant6.stypy_type_store = module_type_store
    simple_if_variant6.stypy_function_name = 'simple_if_variant6'
    simple_if_variant6.stypy_param_names_list = ['a']
    simple_if_variant6.stypy_varargs_param_name = None
    simple_if_variant6.stypy_kwargs_param_name = None
    simple_if_variant6.stypy_call_defaults = defaults
    simple_if_variant6.stypy_call_varargs = varargs
    simple_if_variant6.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_variant6', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_variant6', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_variant6(...)' code ##################

    
    # Assigning a Str to a Name (line 62):
    str_4820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'b', str_4820)
    
    # Type idiom detected: calculating its left and rigth part (line 63)
    # Getting the type of 'a' (line 63)
    a_4821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 14), 'a')
    # Getting the type of 'int' (line 63)
    int_4822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 7), 'int')
    
    (may_be_4823, more_types_in_union_4824) = may_be_type(a_4821, int_4822)

    if may_be_4823:

        if more_types_in_union_4824:
            # Runtime conditional SSA (line 63)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'a', int_4822())
        
        # Assigning a BinOp to a Name (line 64):
        # Getting the type of 'a' (line 64)
        a_4825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'a')
        int_4826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 16), 'int')
        # Applying the binary operator 'div' (line 64)
        result_div_4827 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 12), 'div', a_4825, int_4826)
        
        # Assigning a type to the variable 'r' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'r', result_div_4827)
        
        # Assigning a Subscript to a Name (line 65):
        
        # Obtaining the type of the subscript
        int_4828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 15), 'int')
        # Getting the type of 'a' (line 65)
        a_4829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 65)
        getitem___4830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 13), a_4829, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 65)
        subscript_call_result_4831 = invoke(stypy.reporting.localization.Localization(__file__, 65, 13), getitem___4830, int_4828)
        
        # Assigning a type to the variable 'r2' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'r2', subscript_call_result_4831)
        
        # Assigning a Num to a Name (line 66):
        int_4832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 12), 'int')
        # Assigning a type to the variable 'b' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'b', int_4832)

        if more_types_in_union_4824:
            # SSA join for if statement (line 63)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 67):
    # Getting the type of 'a' (line 67)
    a_4833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 9), 'a')
    int_4834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 13), 'int')
    # Applying the binary operator 'div' (line 67)
    result_div_4835 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 9), 'div', a_4833, int_4834)
    
    # Assigning a type to the variable 'r3' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'r3', result_div_4835)
    
    # Assigning a BinOp to a Name (line 68):
    # Getting the type of 'b' (line 68)
    b_4836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 9), 'b')
    int_4837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 13), 'int')
    # Applying the binary operator 'div' (line 68)
    result_div_4838 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 9), 'div', b_4836, int_4837)
    
    # Assigning a type to the variable 'r4' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'r4', result_div_4838)
    
    # ################# End of 'simple_if_variant6(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_variant6' in the type store
    # Getting the type of 'stypy_return_type' (line 61)
    stypy_return_type_4839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4839)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_variant6'
    return stypy_return_type_4839

# Assigning a type to the variable 'simple_if_variant6' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'simple_if_variant6', simple_if_variant6)

@norecursion
def simple_if_variant7(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_variant7'
    module_type_store = module_type_store.open_function_context('simple_if_variant7', 71, 0, False)
    
    # Passed parameters checking function
    simple_if_variant7.stypy_localization = localization
    simple_if_variant7.stypy_type_of_self = None
    simple_if_variant7.stypy_type_store = module_type_store
    simple_if_variant7.stypy_function_name = 'simple_if_variant7'
    simple_if_variant7.stypy_param_names_list = ['a']
    simple_if_variant7.stypy_varargs_param_name = None
    simple_if_variant7.stypy_kwargs_param_name = None
    simple_if_variant7.stypy_call_defaults = defaults
    simple_if_variant7.stypy_call_varargs = varargs
    simple_if_variant7.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_variant7', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_variant7', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_variant7(...)' code ##################

    
    # Assigning a Str to a Name (line 72):
    str_4840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'b', str_4840)
    
    # Type idiom detected: calculating its left and rigth part (line 73)
    # Getting the type of 'a' (line 73)
    a_4841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 14), 'a')
    # Getting the type of 'int' (line 73)
    int_4842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 7), 'int')
    
    (may_be_4843, more_types_in_union_4844) = may_be_type(a_4841, int_4842)

    if may_be_4843:

        if more_types_in_union_4844:
            # Runtime conditional SSA (line 73)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'a', int_4842())
        
        # Assigning a BinOp to a Name (line 74):
        # Getting the type of 'a' (line 74)
        a_4845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'a')
        int_4846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 16), 'int')
        # Applying the binary operator 'div' (line 74)
        result_div_4847 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 12), 'div', a_4845, int_4846)
        
        # Assigning a type to the variable 'r' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'r', result_div_4847)
        
        # Assigning a Subscript to a Name (line 75):
        
        # Obtaining the type of the subscript
        int_4848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 15), 'int')
        # Getting the type of 'a' (line 75)
        a_4849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 75)
        getitem___4850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 13), a_4849, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 75)
        subscript_call_result_4851 = invoke(stypy.reporting.localization.Localization(__file__, 75, 13), getitem___4850, int_4848)
        
        # Assigning a type to the variable 'r2' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'r2', subscript_call_result_4851)
        
        # Assigning a Num to a Name (line 76):
        int_4852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 12), 'int')
        # Assigning a type to the variable 'b' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'b', int_4852)

        if more_types_in_union_4844:
            # SSA join for if statement (line 73)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 77):
    # Getting the type of 'a' (line 77)
    a_4853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 9), 'a')
    int_4854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 13), 'int')
    # Applying the binary operator 'div' (line 77)
    result_div_4855 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 9), 'div', a_4853, int_4854)
    
    # Assigning a type to the variable 'r3' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'r3', result_div_4855)
    
    # Assigning a BinOp to a Name (line 78):
    # Getting the type of 'b' (line 78)
    b_4856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 9), 'b')
    int_4857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 13), 'int')
    # Applying the binary operator 'div' (line 78)
    result_div_4858 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 9), 'div', b_4856, int_4857)
    
    # Assigning a type to the variable 'r4' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'r4', result_div_4858)
    
    # ################# End of 'simple_if_variant7(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_variant7' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_4859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4859)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_variant7'
    return stypy_return_type_4859

# Assigning a type to the variable 'simple_if_variant7' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'simple_if_variant7', simple_if_variant7)

# Call to simple_if_variant1(...): (line 81)
# Processing the call arguments (line 81)
# Getting the type of 'union' (line 81)
union_4861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 19), 'union', False)
# Processing the call keyword arguments (line 81)
kwargs_4862 = {}
# Getting the type of 'simple_if_variant1' (line 81)
simple_if_variant1_4860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'simple_if_variant1', False)
# Calling simple_if_variant1(args, kwargs) (line 81)
simple_if_variant1_call_result_4863 = invoke(stypy.reporting.localization.Localization(__file__, 81, 0), simple_if_variant1_4860, *[union_4861], **kwargs_4862)


# Call to simple_if_variant2(...): (line 82)
# Processing the call arguments (line 82)
# Getting the type of 'union' (line 82)
union_4865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'union', False)
# Processing the call keyword arguments (line 82)
kwargs_4866 = {}
# Getting the type of 'simple_if_variant2' (line 82)
simple_if_variant2_4864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'simple_if_variant2', False)
# Calling simple_if_variant2(args, kwargs) (line 82)
simple_if_variant2_call_result_4867 = invoke(stypy.reporting.localization.Localization(__file__, 82, 0), simple_if_variant2_4864, *[union_4865], **kwargs_4866)


# Call to simple_if_variant3(...): (line 83)
# Processing the call arguments (line 83)
# Getting the type of 'union' (line 83)
union_4869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'union', False)
# Processing the call keyword arguments (line 83)
kwargs_4870 = {}
# Getting the type of 'simple_if_variant3' (line 83)
simple_if_variant3_4868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'simple_if_variant3', False)
# Calling simple_if_variant3(args, kwargs) (line 83)
simple_if_variant3_call_result_4871 = invoke(stypy.reporting.localization.Localization(__file__, 83, 0), simple_if_variant3_4868, *[union_4869], **kwargs_4870)


# Call to simple_if_variant4(...): (line 84)
# Processing the call arguments (line 84)
# Getting the type of 'union' (line 84)
union_4873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 19), 'union', False)
# Processing the call keyword arguments (line 84)
kwargs_4874 = {}
# Getting the type of 'simple_if_variant4' (line 84)
simple_if_variant4_4872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'simple_if_variant4', False)
# Calling simple_if_variant4(args, kwargs) (line 84)
simple_if_variant4_call_result_4875 = invoke(stypy.reporting.localization.Localization(__file__, 84, 0), simple_if_variant4_4872, *[union_4873], **kwargs_4874)


# Call to simple_if_variant5(...): (line 85)
# Processing the call arguments (line 85)
# Getting the type of 'union' (line 85)
union_4877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 19), 'union', False)
# Processing the call keyword arguments (line 85)
kwargs_4878 = {}
# Getting the type of 'simple_if_variant5' (line 85)
simple_if_variant5_4876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'simple_if_variant5', False)
# Calling simple_if_variant5(args, kwargs) (line 85)
simple_if_variant5_call_result_4879 = invoke(stypy.reporting.localization.Localization(__file__, 85, 0), simple_if_variant5_4876, *[union_4877], **kwargs_4878)


# Call to simple_if_variant6(...): (line 86)
# Processing the call arguments (line 86)
# Getting the type of 'union' (line 86)
union_4881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'union', False)
# Processing the call keyword arguments (line 86)
kwargs_4882 = {}
# Getting the type of 'simple_if_variant6' (line 86)
simple_if_variant6_4880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'simple_if_variant6', False)
# Calling simple_if_variant6(args, kwargs) (line 86)
simple_if_variant6_call_result_4883 = invoke(stypy.reporting.localization.Localization(__file__, 86, 0), simple_if_variant6_4880, *[union_4881], **kwargs_4882)


# Call to simple_if_variant7(...): (line 87)
# Processing the call arguments (line 87)
# Getting the type of 'union' (line 87)
union_4885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), 'union', False)
# Processing the call keyword arguments (line 87)
kwargs_4886 = {}
# Getting the type of 'simple_if_variant7' (line 87)
simple_if_variant7_4884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'simple_if_variant7', False)
# Calling simple_if_variant7(args, kwargs) (line 87)
simple_if_variant7_call_result_4887 = invoke(stypy.reporting.localization.Localization(__file__, 87, 0), simple_if_variant7_4884, *[union_4885], **kwargs_4886)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
