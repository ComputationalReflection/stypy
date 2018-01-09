
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
9: def simple_if_isinstance_base1(a):
10:     b = "hi"
11:     if isinstance(a, int):
12:         r = a / 3
13:         r2 = a[0]
14:         b = 3
15:     r3 = a / 3
16:     r4 = b / 3
17: 
18: 
19: def simple_if_isinstance_base2(a):
20:     b = "hi"
21:     if isinstance(a, int):
22:         r = a / 3
23:         r2 = a[0]
24:         b = 3
25:     r3 = a / 3
26:     r4 = b / 3
27: 
28: 
29: def simple_if_isinstance_base3(a):
30:     b = "hi"
31:     if isinstance(a, int):
32:         r = a / 3
33:         r2 = a[0]
34:         b = 3
35:     r3 = a / 3
36:     r4 = b / 3
37: 
38: 
39: def sum(a, b):
40:     return a + b
41: 
42: 
43: def concat(a, b):
44:     return str(a) + str(b)
45: 
46: 
47: def simple_if_isinstance_call_int(a):
48:     b = "hi"
49:     if isinstance(sum(a, a), int):
50:         r = a / 3
51:         r2 = a[0]
52:         b = 3
53:     r3 = a / 3
54:     r4 = b / 3
55: 
56: 
57: def simple_if_isinstance_call_str(a):
58:     b = "hi"
59:     if not isinstance(concat(a, a), int):
60:         r = a / 3
61:         r2 = a[0]
62:         b = 3
63:     r3 = a / 3
64:     r4 = b / 3
65: 
66: 
67: simple_if_isinstance_base1(theInt)
68: simple_if_isinstance_base2(theStr)
69: simple_if_isinstance_base3(union)
70: 
71: simple_if_isinstance_call_int(theInt)
72: simple_if_isinstance_call_str(union)
73: 
74: 
75: def simple_if_not_isinstance_idiom(a):
76:     b = "hi"
77:     if not isinstance(a, int):
78:         r = a / 3
79:         r2 = a[0]
80:         b = 3
81:     r3 = a / 3
82:     r4 = b / 3
83: 
84: 
85: simple_if_not_isinstance_idiom(union)
86: 
87: 
88: #
89: class Foo:
90:     def __init__(self):
91:         self.attr = 4
92:         self.strattr = "bar"
93: 
94: 
95: def simple_if_isinstance_idiom_attr(a):
96:     b = "hi"
97:     if isinstance(a.attr, int):
98:         r = a.attr / 3
99:         r2 = a.attr[0]
100:         b = 3
101:     r3 = a.attr / 3
102:     r4 = b / 3
103: 
104: 
105: def simple_if_isinstance_idiom_attr_b(a):
106:     b = "hi"
107:     if not isinstance(a.strattr, str):
108:         r = a.attr / 3
109:         r2 = a.strattr[0]
110:         b = 3
111:     r3 = a.strattr / 3
112:     r4 = b / 3
113: 
114: 
115: simple_if_isinstance_idiom_attr(Foo())
116: simple_if_isinstance_idiom_attr_b(Foo())
117: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 1):
int_3627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 9), 'int')
# Assigning a type to the variable 'theInt' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'theInt', int_3627)

# Assigning a Str to a Name (line 2):
str_3628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 9), 'str', 'hi')
# Assigning a type to the variable 'theStr' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'theStr', str_3628)

# Getting the type of 'True' (line 3)
True_3629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 3), 'True')
# Testing the type of an if condition (line 3)
if_condition_3630 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 3, 0), True_3629)
# Assigning a type to the variable 'if_condition_3630' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'if_condition_3630', if_condition_3630)
# SSA begins for if statement (line 3)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Num to a Name (line 4):
int_3631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 12), 'int')
# Assigning a type to the variable 'union' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'union', int_3631)
# SSA branch for the else part of an if statement (line 3)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 6):
str_3632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'str', 'hi')
# Assigning a type to the variable 'union' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'union', str_3632)
# SSA join for if statement (line 3)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def simple_if_isinstance_base1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_isinstance_base1'
    module_type_store = module_type_store.open_function_context('simple_if_isinstance_base1', 9, 0, False)
    
    # Passed parameters checking function
    simple_if_isinstance_base1.stypy_localization = localization
    simple_if_isinstance_base1.stypy_type_of_self = None
    simple_if_isinstance_base1.stypy_type_store = module_type_store
    simple_if_isinstance_base1.stypy_function_name = 'simple_if_isinstance_base1'
    simple_if_isinstance_base1.stypy_param_names_list = ['a']
    simple_if_isinstance_base1.stypy_varargs_param_name = None
    simple_if_isinstance_base1.stypy_kwargs_param_name = None
    simple_if_isinstance_base1.stypy_call_defaults = defaults
    simple_if_isinstance_base1.stypy_call_varargs = varargs
    simple_if_isinstance_base1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_isinstance_base1', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_isinstance_base1', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_isinstance_base1(...)' code ##################

    
    # Assigning a Str to a Name (line 10):
    str_3633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'b', str_3633)
    
    # Type idiom detected: calculating its left and rigth part (line 11)
    # Getting the type of 'int' (line 11)
    int_3634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 21), 'int')
    # Getting the type of 'a' (line 11)
    a_3635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 18), 'a')
    
    (may_be_3636, more_types_in_union_3637) = may_be_subtype(int_3634, a_3635)

    if may_be_3636:

        if more_types_in_union_3637:
            # Runtime conditional SSA (line 11)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'a', remove_not_subtype_from_union(a_3635, int))
        
        # Assigning a BinOp to a Name (line 12):
        # Getting the type of 'a' (line 12)
        a_3638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'a')
        int_3639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 16), 'int')
        # Applying the binary operator 'div' (line 12)
        result_div_3640 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 12), 'div', a_3638, int_3639)
        
        # Assigning a type to the variable 'r' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'r', result_div_3640)
        
        # Assigning a Subscript to a Name (line 13):
        
        # Obtaining the type of the subscript
        int_3641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'int')
        # Getting the type of 'a' (line 13)
        a_3642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 13)
        getitem___3643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 13), a_3642, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 13)
        subscript_call_result_3644 = invoke(stypy.reporting.localization.Localization(__file__, 13, 13), getitem___3643, int_3641)
        
        # Assigning a type to the variable 'r2' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'r2', subscript_call_result_3644)
        
        # Assigning a Num to a Name (line 14):
        int_3645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'int')
        # Assigning a type to the variable 'b' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'b', int_3645)

        if more_types_in_union_3637:
            # SSA join for if statement (line 11)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 15):
    # Getting the type of 'a' (line 15)
    a_3646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 9), 'a')
    int_3647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 13), 'int')
    # Applying the binary operator 'div' (line 15)
    result_div_3648 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 9), 'div', a_3646, int_3647)
    
    # Assigning a type to the variable 'r3' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'r3', result_div_3648)
    
    # Assigning a BinOp to a Name (line 16):
    # Getting the type of 'b' (line 16)
    b_3649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 9), 'b')
    int_3650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'int')
    # Applying the binary operator 'div' (line 16)
    result_div_3651 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 9), 'div', b_3649, int_3650)
    
    # Assigning a type to the variable 'r4' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'r4', result_div_3651)
    
    # ################# End of 'simple_if_isinstance_base1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_isinstance_base1' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_3652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3652)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_isinstance_base1'
    return stypy_return_type_3652

# Assigning a type to the variable 'simple_if_isinstance_base1' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'simple_if_isinstance_base1', simple_if_isinstance_base1)

@norecursion
def simple_if_isinstance_base2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_isinstance_base2'
    module_type_store = module_type_store.open_function_context('simple_if_isinstance_base2', 19, 0, False)
    
    # Passed parameters checking function
    simple_if_isinstance_base2.stypy_localization = localization
    simple_if_isinstance_base2.stypy_type_of_self = None
    simple_if_isinstance_base2.stypy_type_store = module_type_store
    simple_if_isinstance_base2.stypy_function_name = 'simple_if_isinstance_base2'
    simple_if_isinstance_base2.stypy_param_names_list = ['a']
    simple_if_isinstance_base2.stypy_varargs_param_name = None
    simple_if_isinstance_base2.stypy_kwargs_param_name = None
    simple_if_isinstance_base2.stypy_call_defaults = defaults
    simple_if_isinstance_base2.stypy_call_varargs = varargs
    simple_if_isinstance_base2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_isinstance_base2', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_isinstance_base2', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_isinstance_base2(...)' code ##################

    
    # Assigning a Str to a Name (line 20):
    str_3653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'b', str_3653)
    
    # Type idiom detected: calculating its left and rigth part (line 21)
    # Getting the type of 'int' (line 21)
    int_3654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 21), 'int')
    # Getting the type of 'a' (line 21)
    a_3655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 18), 'a')
    
    (may_be_3656, more_types_in_union_3657) = may_be_subtype(int_3654, a_3655)

    if may_be_3656:

        if more_types_in_union_3657:
            # Runtime conditional SSA (line 21)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'a', remove_not_subtype_from_union(a_3655, int))
        
        # Assigning a BinOp to a Name (line 22):
        # Getting the type of 'a' (line 22)
        a_3658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'a')
        int_3659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'int')
        # Applying the binary operator 'div' (line 22)
        result_div_3660 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 12), 'div', a_3658, int_3659)
        
        # Assigning a type to the variable 'r' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'r', result_div_3660)
        
        # Assigning a Subscript to a Name (line 23):
        
        # Obtaining the type of the subscript
        int_3661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'int')
        # Getting the type of 'a' (line 23)
        a_3662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 23)
        getitem___3663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 13), a_3662, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 23)
        subscript_call_result_3664 = invoke(stypy.reporting.localization.Localization(__file__, 23, 13), getitem___3663, int_3661)
        
        # Assigning a type to the variable 'r2' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'r2', subscript_call_result_3664)
        
        # Assigning a Num to a Name (line 24):
        int_3665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 12), 'int')
        # Assigning a type to the variable 'b' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'b', int_3665)

        if more_types_in_union_3657:
            # SSA join for if statement (line 21)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 25):
    # Getting the type of 'a' (line 25)
    a_3666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 9), 'a')
    int_3667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'int')
    # Applying the binary operator 'div' (line 25)
    result_div_3668 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 9), 'div', a_3666, int_3667)
    
    # Assigning a type to the variable 'r3' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'r3', result_div_3668)
    
    # Assigning a BinOp to a Name (line 26):
    # Getting the type of 'b' (line 26)
    b_3669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 9), 'b')
    int_3670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 13), 'int')
    # Applying the binary operator 'div' (line 26)
    result_div_3671 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 9), 'div', b_3669, int_3670)
    
    # Assigning a type to the variable 'r4' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'r4', result_div_3671)
    
    # ################# End of 'simple_if_isinstance_base2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_isinstance_base2' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_3672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3672)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_isinstance_base2'
    return stypy_return_type_3672

# Assigning a type to the variable 'simple_if_isinstance_base2' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'simple_if_isinstance_base2', simple_if_isinstance_base2)

@norecursion
def simple_if_isinstance_base3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_isinstance_base3'
    module_type_store = module_type_store.open_function_context('simple_if_isinstance_base3', 29, 0, False)
    
    # Passed parameters checking function
    simple_if_isinstance_base3.stypy_localization = localization
    simple_if_isinstance_base3.stypy_type_of_self = None
    simple_if_isinstance_base3.stypy_type_store = module_type_store
    simple_if_isinstance_base3.stypy_function_name = 'simple_if_isinstance_base3'
    simple_if_isinstance_base3.stypy_param_names_list = ['a']
    simple_if_isinstance_base3.stypy_varargs_param_name = None
    simple_if_isinstance_base3.stypy_kwargs_param_name = None
    simple_if_isinstance_base3.stypy_call_defaults = defaults
    simple_if_isinstance_base3.stypy_call_varargs = varargs
    simple_if_isinstance_base3.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_isinstance_base3', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_isinstance_base3', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_isinstance_base3(...)' code ##################

    
    # Assigning a Str to a Name (line 30):
    str_3673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'b', str_3673)
    
    # Type idiom detected: calculating its left and rigth part (line 31)
    # Getting the type of 'int' (line 31)
    int_3674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 21), 'int')
    # Getting the type of 'a' (line 31)
    a_3675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 18), 'a')
    
    (may_be_3676, more_types_in_union_3677) = may_be_subtype(int_3674, a_3675)

    if may_be_3676:

        if more_types_in_union_3677:
            # Runtime conditional SSA (line 31)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'a', remove_not_subtype_from_union(a_3675, int))
        
        # Assigning a BinOp to a Name (line 32):
        # Getting the type of 'a' (line 32)
        a_3678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'a')
        int_3679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 16), 'int')
        # Applying the binary operator 'div' (line 32)
        result_div_3680 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 12), 'div', a_3678, int_3679)
        
        # Assigning a type to the variable 'r' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'r', result_div_3680)
        
        # Assigning a Subscript to a Name (line 33):
        
        # Obtaining the type of the subscript
        int_3681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 15), 'int')
        # Getting the type of 'a' (line 33)
        a_3682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 33)
        getitem___3683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 13), a_3682, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 33)
        subscript_call_result_3684 = invoke(stypy.reporting.localization.Localization(__file__, 33, 13), getitem___3683, int_3681)
        
        # Assigning a type to the variable 'r2' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'r2', subscript_call_result_3684)
        
        # Assigning a Num to a Name (line 34):
        int_3685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 12), 'int')
        # Assigning a type to the variable 'b' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'b', int_3685)

        if more_types_in_union_3677:
            # SSA join for if statement (line 31)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 35):
    # Getting the type of 'a' (line 35)
    a_3686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 9), 'a')
    int_3687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 13), 'int')
    # Applying the binary operator 'div' (line 35)
    result_div_3688 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 9), 'div', a_3686, int_3687)
    
    # Assigning a type to the variable 'r3' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'r3', result_div_3688)
    
    # Assigning a BinOp to a Name (line 36):
    # Getting the type of 'b' (line 36)
    b_3689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 9), 'b')
    int_3690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 13), 'int')
    # Applying the binary operator 'div' (line 36)
    result_div_3691 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 9), 'div', b_3689, int_3690)
    
    # Assigning a type to the variable 'r4' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'r4', result_div_3691)
    
    # ################# End of 'simple_if_isinstance_base3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_isinstance_base3' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_3692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3692)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_isinstance_base3'
    return stypy_return_type_3692

# Assigning a type to the variable 'simple_if_isinstance_base3' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'simple_if_isinstance_base3', simple_if_isinstance_base3)

@norecursion
def sum(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sum'
    module_type_store = module_type_store.open_function_context('sum', 39, 0, False)
    
    # Passed parameters checking function
    sum.stypy_localization = localization
    sum.stypy_type_of_self = None
    sum.stypy_type_store = module_type_store
    sum.stypy_function_name = 'sum'
    sum.stypy_param_names_list = ['a', 'b']
    sum.stypy_varargs_param_name = None
    sum.stypy_kwargs_param_name = None
    sum.stypy_call_defaults = defaults
    sum.stypy_call_varargs = varargs
    sum.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sum', ['a', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sum', localization, ['a', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sum(...)' code ##################

    # Getting the type of 'a' (line 40)
    a_3693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'a')
    # Getting the type of 'b' (line 40)
    b_3694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'b')
    # Applying the binary operator '+' (line 40)
    result_add_3695 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 11), '+', a_3693, b_3694)
    
    # Assigning a type to the variable 'stypy_return_type' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type', result_add_3695)
    
    # ################# End of 'sum(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sum' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_3696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3696)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sum'
    return stypy_return_type_3696

# Assigning a type to the variable 'sum' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'sum', sum)

@norecursion
def concat(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'concat'
    module_type_store = module_type_store.open_function_context('concat', 43, 0, False)
    
    # Passed parameters checking function
    concat.stypy_localization = localization
    concat.stypy_type_of_self = None
    concat.stypy_type_store = module_type_store
    concat.stypy_function_name = 'concat'
    concat.stypy_param_names_list = ['a', 'b']
    concat.stypy_varargs_param_name = None
    concat.stypy_kwargs_param_name = None
    concat.stypy_call_defaults = defaults
    concat.stypy_call_varargs = varargs
    concat.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'concat', ['a', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'concat', localization, ['a', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'concat(...)' code ##################

    
    # Call to str(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'a' (line 44)
    a_3698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'a', False)
    # Processing the call keyword arguments (line 44)
    kwargs_3699 = {}
    # Getting the type of 'str' (line 44)
    str_3697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'str', False)
    # Calling str(args, kwargs) (line 44)
    str_call_result_3700 = invoke(stypy.reporting.localization.Localization(__file__, 44, 11), str_3697, *[a_3698], **kwargs_3699)
    
    
    # Call to str(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'b' (line 44)
    b_3702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 24), 'b', False)
    # Processing the call keyword arguments (line 44)
    kwargs_3703 = {}
    # Getting the type of 'str' (line 44)
    str_3701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'str', False)
    # Calling str(args, kwargs) (line 44)
    str_call_result_3704 = invoke(stypy.reporting.localization.Localization(__file__, 44, 20), str_3701, *[b_3702], **kwargs_3703)
    
    # Applying the binary operator '+' (line 44)
    result_add_3705 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 11), '+', str_call_result_3700, str_call_result_3704)
    
    # Assigning a type to the variable 'stypy_return_type' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type', result_add_3705)
    
    # ################# End of 'concat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'concat' in the type store
    # Getting the type of 'stypy_return_type' (line 43)
    stypy_return_type_3706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3706)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'concat'
    return stypy_return_type_3706

# Assigning a type to the variable 'concat' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'concat', concat)

@norecursion
def simple_if_isinstance_call_int(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_isinstance_call_int'
    module_type_store = module_type_store.open_function_context('simple_if_isinstance_call_int', 47, 0, False)
    
    # Passed parameters checking function
    simple_if_isinstance_call_int.stypy_localization = localization
    simple_if_isinstance_call_int.stypy_type_of_self = None
    simple_if_isinstance_call_int.stypy_type_store = module_type_store
    simple_if_isinstance_call_int.stypy_function_name = 'simple_if_isinstance_call_int'
    simple_if_isinstance_call_int.stypy_param_names_list = ['a']
    simple_if_isinstance_call_int.stypy_varargs_param_name = None
    simple_if_isinstance_call_int.stypy_kwargs_param_name = None
    simple_if_isinstance_call_int.stypy_call_defaults = defaults
    simple_if_isinstance_call_int.stypy_call_varargs = varargs
    simple_if_isinstance_call_int.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_isinstance_call_int', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_isinstance_call_int', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_isinstance_call_int(...)' code ##################

    
    # Assigning a Str to a Name (line 48):
    str_3707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'b', str_3707)
    
    # Type idiom detected: calculating its left and rigth part (line 49)
    # Getting the type of 'int' (line 49)
    int_3708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 29), 'int')
    
    # Call to sum(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'a' (line 49)
    a_3710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 22), 'a', False)
    # Getting the type of 'a' (line 49)
    a_3711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'a', False)
    # Processing the call keyword arguments (line 49)
    kwargs_3712 = {}
    # Getting the type of 'sum' (line 49)
    sum_3709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 18), 'sum', False)
    # Calling sum(args, kwargs) (line 49)
    sum_call_result_3713 = invoke(stypy.reporting.localization.Localization(__file__, 49, 18), sum_3709, *[a_3710, a_3711], **kwargs_3712)
    
    
    (may_be_3714, more_types_in_union_3715) = may_be_subtype(int_3708, sum_call_result_3713)

    if may_be_3714:

        if more_types_in_union_3715:
            # Runtime conditional SSA (line 49)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 50):
        # Getting the type of 'a' (line 50)
        a_3716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'a')
        int_3717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 16), 'int')
        # Applying the binary operator 'div' (line 50)
        result_div_3718 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 12), 'div', a_3716, int_3717)
        
        # Assigning a type to the variable 'r' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'r', result_div_3718)
        
        # Assigning a Subscript to a Name (line 51):
        
        # Obtaining the type of the subscript
        int_3719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 15), 'int')
        # Getting the type of 'a' (line 51)
        a_3720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___3721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 13), a_3720, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_3722 = invoke(stypy.reporting.localization.Localization(__file__, 51, 13), getitem___3721, int_3719)
        
        # Assigning a type to the variable 'r2' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'r2', subscript_call_result_3722)
        
        # Assigning a Num to a Name (line 52):
        int_3723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 12), 'int')
        # Assigning a type to the variable 'b' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'b', int_3723)

        if more_types_in_union_3715:
            # SSA join for if statement (line 49)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 53):
    # Getting the type of 'a' (line 53)
    a_3724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 9), 'a')
    int_3725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 13), 'int')
    # Applying the binary operator 'div' (line 53)
    result_div_3726 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 9), 'div', a_3724, int_3725)
    
    # Assigning a type to the variable 'r3' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'r3', result_div_3726)
    
    # Assigning a BinOp to a Name (line 54):
    # Getting the type of 'b' (line 54)
    b_3727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 9), 'b')
    int_3728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 13), 'int')
    # Applying the binary operator 'div' (line 54)
    result_div_3729 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 9), 'div', b_3727, int_3728)
    
    # Assigning a type to the variable 'r4' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'r4', result_div_3729)
    
    # ################# End of 'simple_if_isinstance_call_int(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_isinstance_call_int' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_3730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3730)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_isinstance_call_int'
    return stypy_return_type_3730

# Assigning a type to the variable 'simple_if_isinstance_call_int' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'simple_if_isinstance_call_int', simple_if_isinstance_call_int)

@norecursion
def simple_if_isinstance_call_str(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_isinstance_call_str'
    module_type_store = module_type_store.open_function_context('simple_if_isinstance_call_str', 57, 0, False)
    
    # Passed parameters checking function
    simple_if_isinstance_call_str.stypy_localization = localization
    simple_if_isinstance_call_str.stypy_type_of_self = None
    simple_if_isinstance_call_str.stypy_type_store = module_type_store
    simple_if_isinstance_call_str.stypy_function_name = 'simple_if_isinstance_call_str'
    simple_if_isinstance_call_str.stypy_param_names_list = ['a']
    simple_if_isinstance_call_str.stypy_varargs_param_name = None
    simple_if_isinstance_call_str.stypy_kwargs_param_name = None
    simple_if_isinstance_call_str.stypy_call_defaults = defaults
    simple_if_isinstance_call_str.stypy_call_varargs = varargs
    simple_if_isinstance_call_str.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_isinstance_call_str', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_isinstance_call_str', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_isinstance_call_str(...)' code ##################

    
    # Assigning a Str to a Name (line 58):
    str_3731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'b', str_3731)
    
    # Type idiom detected: calculating its left and rigth part (line 59)
    # Getting the type of 'int' (line 59)
    int_3732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 36), 'int')
    
    # Call to concat(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'a' (line 59)
    a_3734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 29), 'a', False)
    # Getting the type of 'a' (line 59)
    a_3735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 32), 'a', False)
    # Processing the call keyword arguments (line 59)
    kwargs_3736 = {}
    # Getting the type of 'concat' (line 59)
    concat_3733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 22), 'concat', False)
    # Calling concat(args, kwargs) (line 59)
    concat_call_result_3737 = invoke(stypy.reporting.localization.Localization(__file__, 59, 22), concat_3733, *[a_3734, a_3735], **kwargs_3736)
    
    
    (may_be_3738, more_types_in_union_3739) = may_not_be_subtype(int_3732, concat_call_result_3737)

    if may_be_3738:

        if more_types_in_union_3739:
            # Runtime conditional SSA (line 59)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 60):
        # Getting the type of 'a' (line 60)
        a_3740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'a')
        int_3741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 16), 'int')
        # Applying the binary operator 'div' (line 60)
        result_div_3742 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 12), 'div', a_3740, int_3741)
        
        # Assigning a type to the variable 'r' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'r', result_div_3742)
        
        # Assigning a Subscript to a Name (line 61):
        
        # Obtaining the type of the subscript
        int_3743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 15), 'int')
        # Getting the type of 'a' (line 61)
        a_3744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___3745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 13), a_3744, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_3746 = invoke(stypy.reporting.localization.Localization(__file__, 61, 13), getitem___3745, int_3743)
        
        # Assigning a type to the variable 'r2' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'r2', subscript_call_result_3746)
        
        # Assigning a Num to a Name (line 62):
        int_3747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 12), 'int')
        # Assigning a type to the variable 'b' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'b', int_3747)

        if more_types_in_union_3739:
            # SSA join for if statement (line 59)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 63):
    # Getting the type of 'a' (line 63)
    a_3748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 9), 'a')
    int_3749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 13), 'int')
    # Applying the binary operator 'div' (line 63)
    result_div_3750 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 9), 'div', a_3748, int_3749)
    
    # Assigning a type to the variable 'r3' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'r3', result_div_3750)
    
    # Assigning a BinOp to a Name (line 64):
    # Getting the type of 'b' (line 64)
    b_3751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 9), 'b')
    int_3752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 13), 'int')
    # Applying the binary operator 'div' (line 64)
    result_div_3753 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 9), 'div', b_3751, int_3752)
    
    # Assigning a type to the variable 'r4' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'r4', result_div_3753)
    
    # ################# End of 'simple_if_isinstance_call_str(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_isinstance_call_str' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_3754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3754)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_isinstance_call_str'
    return stypy_return_type_3754

# Assigning a type to the variable 'simple_if_isinstance_call_str' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'simple_if_isinstance_call_str', simple_if_isinstance_call_str)

# Call to simple_if_isinstance_base1(...): (line 67)
# Processing the call arguments (line 67)
# Getting the type of 'theInt' (line 67)
theInt_3756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 27), 'theInt', False)
# Processing the call keyword arguments (line 67)
kwargs_3757 = {}
# Getting the type of 'simple_if_isinstance_base1' (line 67)
simple_if_isinstance_base1_3755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'simple_if_isinstance_base1', False)
# Calling simple_if_isinstance_base1(args, kwargs) (line 67)
simple_if_isinstance_base1_call_result_3758 = invoke(stypy.reporting.localization.Localization(__file__, 67, 0), simple_if_isinstance_base1_3755, *[theInt_3756], **kwargs_3757)


# Call to simple_if_isinstance_base2(...): (line 68)
# Processing the call arguments (line 68)
# Getting the type of 'theStr' (line 68)
theStr_3760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 27), 'theStr', False)
# Processing the call keyword arguments (line 68)
kwargs_3761 = {}
# Getting the type of 'simple_if_isinstance_base2' (line 68)
simple_if_isinstance_base2_3759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'simple_if_isinstance_base2', False)
# Calling simple_if_isinstance_base2(args, kwargs) (line 68)
simple_if_isinstance_base2_call_result_3762 = invoke(stypy.reporting.localization.Localization(__file__, 68, 0), simple_if_isinstance_base2_3759, *[theStr_3760], **kwargs_3761)


# Call to simple_if_isinstance_base3(...): (line 69)
# Processing the call arguments (line 69)
# Getting the type of 'union' (line 69)
union_3764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 27), 'union', False)
# Processing the call keyword arguments (line 69)
kwargs_3765 = {}
# Getting the type of 'simple_if_isinstance_base3' (line 69)
simple_if_isinstance_base3_3763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'simple_if_isinstance_base3', False)
# Calling simple_if_isinstance_base3(args, kwargs) (line 69)
simple_if_isinstance_base3_call_result_3766 = invoke(stypy.reporting.localization.Localization(__file__, 69, 0), simple_if_isinstance_base3_3763, *[union_3764], **kwargs_3765)


# Call to simple_if_isinstance_call_int(...): (line 71)
# Processing the call arguments (line 71)
# Getting the type of 'theInt' (line 71)
theInt_3768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 30), 'theInt', False)
# Processing the call keyword arguments (line 71)
kwargs_3769 = {}
# Getting the type of 'simple_if_isinstance_call_int' (line 71)
simple_if_isinstance_call_int_3767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'simple_if_isinstance_call_int', False)
# Calling simple_if_isinstance_call_int(args, kwargs) (line 71)
simple_if_isinstance_call_int_call_result_3770 = invoke(stypy.reporting.localization.Localization(__file__, 71, 0), simple_if_isinstance_call_int_3767, *[theInt_3768], **kwargs_3769)


# Call to simple_if_isinstance_call_str(...): (line 72)
# Processing the call arguments (line 72)
# Getting the type of 'union' (line 72)
union_3772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 30), 'union', False)
# Processing the call keyword arguments (line 72)
kwargs_3773 = {}
# Getting the type of 'simple_if_isinstance_call_str' (line 72)
simple_if_isinstance_call_str_3771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'simple_if_isinstance_call_str', False)
# Calling simple_if_isinstance_call_str(args, kwargs) (line 72)
simple_if_isinstance_call_str_call_result_3774 = invoke(stypy.reporting.localization.Localization(__file__, 72, 0), simple_if_isinstance_call_str_3771, *[union_3772], **kwargs_3773)


@norecursion
def simple_if_not_isinstance_idiom(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_not_isinstance_idiom'
    module_type_store = module_type_store.open_function_context('simple_if_not_isinstance_idiom', 75, 0, False)
    
    # Passed parameters checking function
    simple_if_not_isinstance_idiom.stypy_localization = localization
    simple_if_not_isinstance_idiom.stypy_type_of_self = None
    simple_if_not_isinstance_idiom.stypy_type_store = module_type_store
    simple_if_not_isinstance_idiom.stypy_function_name = 'simple_if_not_isinstance_idiom'
    simple_if_not_isinstance_idiom.stypy_param_names_list = ['a']
    simple_if_not_isinstance_idiom.stypy_varargs_param_name = None
    simple_if_not_isinstance_idiom.stypy_kwargs_param_name = None
    simple_if_not_isinstance_idiom.stypy_call_defaults = defaults
    simple_if_not_isinstance_idiom.stypy_call_varargs = varargs
    simple_if_not_isinstance_idiom.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_not_isinstance_idiom', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_not_isinstance_idiom', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_not_isinstance_idiom(...)' code ##################

    
    # Assigning a Str to a Name (line 76):
    str_3775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'b', str_3775)
    
    # Type idiom detected: calculating its left and rigth part (line 77)
    # Getting the type of 'int' (line 77)
    int_3776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), 'int')
    # Getting the type of 'a' (line 77)
    a_3777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 22), 'a')
    
    (may_be_3778, more_types_in_union_3779) = may_not_be_subtype(int_3776, a_3777)

    if may_be_3778:

        if more_types_in_union_3779:
            # Runtime conditional SSA (line 77)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'a', remove_subtype_from_union(a_3777, int))
        
        # Assigning a BinOp to a Name (line 78):
        # Getting the type of 'a' (line 78)
        a_3780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'a')
        int_3781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 16), 'int')
        # Applying the binary operator 'div' (line 78)
        result_div_3782 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 12), 'div', a_3780, int_3781)
        
        # Assigning a type to the variable 'r' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'r', result_div_3782)
        
        # Assigning a Subscript to a Name (line 79):
        
        # Obtaining the type of the subscript
        int_3783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 15), 'int')
        # Getting the type of 'a' (line 79)
        a_3784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 79)
        getitem___3785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 13), a_3784, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 79)
        subscript_call_result_3786 = invoke(stypy.reporting.localization.Localization(__file__, 79, 13), getitem___3785, int_3783)
        
        # Assigning a type to the variable 'r2' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'r2', subscript_call_result_3786)
        
        # Assigning a Num to a Name (line 80):
        int_3787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 12), 'int')
        # Assigning a type to the variable 'b' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'b', int_3787)

        if more_types_in_union_3779:
            # SSA join for if statement (line 77)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 81):
    # Getting the type of 'a' (line 81)
    a_3788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 9), 'a')
    int_3789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 13), 'int')
    # Applying the binary operator 'div' (line 81)
    result_div_3790 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 9), 'div', a_3788, int_3789)
    
    # Assigning a type to the variable 'r3' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'r3', result_div_3790)
    
    # Assigning a BinOp to a Name (line 82):
    # Getting the type of 'b' (line 82)
    b_3791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 9), 'b')
    int_3792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 13), 'int')
    # Applying the binary operator 'div' (line 82)
    result_div_3793 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 9), 'div', b_3791, int_3792)
    
    # Assigning a type to the variable 'r4' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'r4', result_div_3793)
    
    # ################# End of 'simple_if_not_isinstance_idiom(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_not_isinstance_idiom' in the type store
    # Getting the type of 'stypy_return_type' (line 75)
    stypy_return_type_3794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3794)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_not_isinstance_idiom'
    return stypy_return_type_3794

# Assigning a type to the variable 'simple_if_not_isinstance_idiom' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'simple_if_not_isinstance_idiom', simple_if_not_isinstance_idiom)

# Call to simple_if_not_isinstance_idiom(...): (line 85)
# Processing the call arguments (line 85)
# Getting the type of 'union' (line 85)
union_3796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 31), 'union', False)
# Processing the call keyword arguments (line 85)
kwargs_3797 = {}
# Getting the type of 'simple_if_not_isinstance_idiom' (line 85)
simple_if_not_isinstance_idiom_3795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'simple_if_not_isinstance_idiom', False)
# Calling simple_if_not_isinstance_idiom(args, kwargs) (line 85)
simple_if_not_isinstance_idiom_call_result_3798 = invoke(stypy.reporting.localization.Localization(__file__, 85, 0), simple_if_not_isinstance_idiom_3795, *[union_3796], **kwargs_3797)

# Declaration of the 'Foo' class

class Foo:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 90, 4, False)
        # Assigning a type to the variable 'self' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Num to a Attribute (line 91):
        int_3799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 20), 'int')
        # Getting the type of 'self' (line 91)
        self_3800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'self')
        # Setting the type of the member 'attr' of a type (line 91)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), self_3800, 'attr', int_3799)
        
        # Assigning a Str to a Attribute (line 92):
        str_3801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 23), 'str', 'bar')
        # Getting the type of 'self' (line 92)
        self_3802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'self')
        # Setting the type of the member 'strattr' of a type (line 92)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), self_3802, 'strattr', str_3801)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Foo' (line 89)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'Foo', Foo)

@norecursion
def simple_if_isinstance_idiom_attr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_isinstance_idiom_attr'
    module_type_store = module_type_store.open_function_context('simple_if_isinstance_idiom_attr', 95, 0, False)
    
    # Passed parameters checking function
    simple_if_isinstance_idiom_attr.stypy_localization = localization
    simple_if_isinstance_idiom_attr.stypy_type_of_self = None
    simple_if_isinstance_idiom_attr.stypy_type_store = module_type_store
    simple_if_isinstance_idiom_attr.stypy_function_name = 'simple_if_isinstance_idiom_attr'
    simple_if_isinstance_idiom_attr.stypy_param_names_list = ['a']
    simple_if_isinstance_idiom_attr.stypy_varargs_param_name = None
    simple_if_isinstance_idiom_attr.stypy_kwargs_param_name = None
    simple_if_isinstance_idiom_attr.stypy_call_defaults = defaults
    simple_if_isinstance_idiom_attr.stypy_call_varargs = varargs
    simple_if_isinstance_idiom_attr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_isinstance_idiom_attr', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_isinstance_idiom_attr', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_isinstance_idiom_attr(...)' code ##################

    
    # Assigning a Str to a Name (line 96):
    str_3803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'b', str_3803)
    
    # Type idiom detected: calculating its left and rigth part (line 97)
    # Getting the type of 'int' (line 97)
    int_3804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 26), 'int')
    # Getting the type of 'a' (line 97)
    a_3805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 18), 'a')
    # Obtaining the member 'attr' of a type (line 97)
    attr_3806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 18), a_3805, 'attr')
    
    (may_be_3807, more_types_in_union_3808) = may_be_subtype(int_3804, attr_3806)

    if may_be_3807:

        if more_types_in_union_3808:
            # Runtime conditional SSA (line 97)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 97)
        a_3809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'a')
        # Obtaining the member 'attr' of a type (line 97)
        attr_3810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 4), a_3809, 'attr')
        # Setting the type of the member 'attr' of a type (line 97)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 4), a_3809, 'attr', remove_not_subtype_from_union(attr_3806, int))
        
        # Assigning a BinOp to a Name (line 98):
        # Getting the type of 'a' (line 98)
        a_3811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'a')
        # Obtaining the member 'attr' of a type (line 98)
        attr_3812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), a_3811, 'attr')
        int_3813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 21), 'int')
        # Applying the binary operator 'div' (line 98)
        result_div_3814 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 12), 'div', attr_3812, int_3813)
        
        # Assigning a type to the variable 'r' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'r', result_div_3814)
        
        # Assigning a Subscript to a Name (line 99):
        
        # Obtaining the type of the subscript
        int_3815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 20), 'int')
        # Getting the type of 'a' (line 99)
        a_3816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 13), 'a')
        # Obtaining the member 'attr' of a type (line 99)
        attr_3817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 13), a_3816, 'attr')
        # Obtaining the member '__getitem__' of a type (line 99)
        getitem___3818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 13), attr_3817, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 99)
        subscript_call_result_3819 = invoke(stypy.reporting.localization.Localization(__file__, 99, 13), getitem___3818, int_3815)
        
        # Assigning a type to the variable 'r2' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'r2', subscript_call_result_3819)
        
        # Assigning a Num to a Name (line 100):
        int_3820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 12), 'int')
        # Assigning a type to the variable 'b' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'b', int_3820)

        if more_types_in_union_3808:
            # SSA join for if statement (line 97)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 101):
    # Getting the type of 'a' (line 101)
    a_3821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 9), 'a')
    # Obtaining the member 'attr' of a type (line 101)
    attr_3822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 9), a_3821, 'attr')
    int_3823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 18), 'int')
    # Applying the binary operator 'div' (line 101)
    result_div_3824 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 9), 'div', attr_3822, int_3823)
    
    # Assigning a type to the variable 'r3' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'r3', result_div_3824)
    
    # Assigning a BinOp to a Name (line 102):
    # Getting the type of 'b' (line 102)
    b_3825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 9), 'b')
    int_3826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 13), 'int')
    # Applying the binary operator 'div' (line 102)
    result_div_3827 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 9), 'div', b_3825, int_3826)
    
    # Assigning a type to the variable 'r4' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'r4', result_div_3827)
    
    # ################# End of 'simple_if_isinstance_idiom_attr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_isinstance_idiom_attr' in the type store
    # Getting the type of 'stypy_return_type' (line 95)
    stypy_return_type_3828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3828)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_isinstance_idiom_attr'
    return stypy_return_type_3828

# Assigning a type to the variable 'simple_if_isinstance_idiom_attr' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'simple_if_isinstance_idiom_attr', simple_if_isinstance_idiom_attr)

@norecursion
def simple_if_isinstance_idiom_attr_b(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_isinstance_idiom_attr_b'
    module_type_store = module_type_store.open_function_context('simple_if_isinstance_idiom_attr_b', 105, 0, False)
    
    # Passed parameters checking function
    simple_if_isinstance_idiom_attr_b.stypy_localization = localization
    simple_if_isinstance_idiom_attr_b.stypy_type_of_self = None
    simple_if_isinstance_idiom_attr_b.stypy_type_store = module_type_store
    simple_if_isinstance_idiom_attr_b.stypy_function_name = 'simple_if_isinstance_idiom_attr_b'
    simple_if_isinstance_idiom_attr_b.stypy_param_names_list = ['a']
    simple_if_isinstance_idiom_attr_b.stypy_varargs_param_name = None
    simple_if_isinstance_idiom_attr_b.stypy_kwargs_param_name = None
    simple_if_isinstance_idiom_attr_b.stypy_call_defaults = defaults
    simple_if_isinstance_idiom_attr_b.stypy_call_varargs = varargs
    simple_if_isinstance_idiom_attr_b.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_isinstance_idiom_attr_b', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_isinstance_idiom_attr_b', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_isinstance_idiom_attr_b(...)' code ##################

    
    # Assigning a Str to a Name (line 106):
    str_3829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'b', str_3829)
    
    # Type idiom detected: calculating its left and rigth part (line 107)
    # Getting the type of 'str' (line 107)
    str_3830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 33), 'str')
    # Getting the type of 'a' (line 107)
    a_3831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 22), 'a')
    # Obtaining the member 'strattr' of a type (line 107)
    strattr_3832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 22), a_3831, 'strattr')
    
    (may_be_3833, more_types_in_union_3834) = may_not_be_subtype(str_3830, strattr_3832)

    if may_be_3833:

        if more_types_in_union_3834:
            # Runtime conditional SSA (line 107)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 107)
        a_3835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'a')
        # Obtaining the member 'strattr' of a type (line 107)
        strattr_3836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 4), a_3835, 'strattr')
        # Setting the type of the member 'strattr' of a type (line 107)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 4), a_3835, 'strattr', remove_subtype_from_union(strattr_3832, str))
        
        # Assigning a BinOp to a Name (line 108):
        # Getting the type of 'a' (line 108)
        a_3837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'a')
        # Obtaining the member 'attr' of a type (line 108)
        attr_3838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), a_3837, 'attr')
        int_3839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 21), 'int')
        # Applying the binary operator 'div' (line 108)
        result_div_3840 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 12), 'div', attr_3838, int_3839)
        
        # Assigning a type to the variable 'r' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'r', result_div_3840)
        
        # Assigning a Subscript to a Name (line 109):
        
        # Obtaining the type of the subscript
        int_3841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 23), 'int')
        # Getting the type of 'a' (line 109)
        a_3842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 13), 'a')
        # Obtaining the member 'strattr' of a type (line 109)
        strattr_3843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 13), a_3842, 'strattr')
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___3844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 13), strattr_3843, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_3845 = invoke(stypy.reporting.localization.Localization(__file__, 109, 13), getitem___3844, int_3841)
        
        # Assigning a type to the variable 'r2' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'r2', subscript_call_result_3845)
        
        # Assigning a Num to a Name (line 110):
        int_3846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 12), 'int')
        # Assigning a type to the variable 'b' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'b', int_3846)

        if more_types_in_union_3834:
            # SSA join for if statement (line 107)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 111):
    # Getting the type of 'a' (line 111)
    a_3847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 9), 'a')
    # Obtaining the member 'strattr' of a type (line 111)
    strattr_3848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 9), a_3847, 'strattr')
    int_3849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 21), 'int')
    # Applying the binary operator 'div' (line 111)
    result_div_3850 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 9), 'div', strattr_3848, int_3849)
    
    # Assigning a type to the variable 'r3' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'r3', result_div_3850)
    
    # Assigning a BinOp to a Name (line 112):
    # Getting the type of 'b' (line 112)
    b_3851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 9), 'b')
    int_3852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 13), 'int')
    # Applying the binary operator 'div' (line 112)
    result_div_3853 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 9), 'div', b_3851, int_3852)
    
    # Assigning a type to the variable 'r4' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'r4', result_div_3853)
    
    # ################# End of 'simple_if_isinstance_idiom_attr_b(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_isinstance_idiom_attr_b' in the type store
    # Getting the type of 'stypy_return_type' (line 105)
    stypy_return_type_3854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3854)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_isinstance_idiom_attr_b'
    return stypy_return_type_3854

# Assigning a type to the variable 'simple_if_isinstance_idiom_attr_b' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'simple_if_isinstance_idiom_attr_b', simple_if_isinstance_idiom_attr_b)

# Call to simple_if_isinstance_idiom_attr(...): (line 115)
# Processing the call arguments (line 115)

# Call to Foo(...): (line 115)
# Processing the call keyword arguments (line 115)
kwargs_3857 = {}
# Getting the type of 'Foo' (line 115)
Foo_3856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 32), 'Foo', False)
# Calling Foo(args, kwargs) (line 115)
Foo_call_result_3858 = invoke(stypy.reporting.localization.Localization(__file__, 115, 32), Foo_3856, *[], **kwargs_3857)

# Processing the call keyword arguments (line 115)
kwargs_3859 = {}
# Getting the type of 'simple_if_isinstance_idiom_attr' (line 115)
simple_if_isinstance_idiom_attr_3855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'simple_if_isinstance_idiom_attr', False)
# Calling simple_if_isinstance_idiom_attr(args, kwargs) (line 115)
simple_if_isinstance_idiom_attr_call_result_3860 = invoke(stypy.reporting.localization.Localization(__file__, 115, 0), simple_if_isinstance_idiom_attr_3855, *[Foo_call_result_3858], **kwargs_3859)


# Call to simple_if_isinstance_idiom_attr_b(...): (line 116)
# Processing the call arguments (line 116)

# Call to Foo(...): (line 116)
# Processing the call keyword arguments (line 116)
kwargs_3863 = {}
# Getting the type of 'Foo' (line 116)
Foo_3862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 34), 'Foo', False)
# Calling Foo(args, kwargs) (line 116)
Foo_call_result_3864 = invoke(stypy.reporting.localization.Localization(__file__, 116, 34), Foo_3862, *[], **kwargs_3863)

# Processing the call keyword arguments (line 116)
kwargs_3865 = {}
# Getting the type of 'simple_if_isinstance_idiom_attr_b' (line 116)
simple_if_isinstance_idiom_attr_b_3861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'simple_if_isinstance_idiom_attr_b', False)
# Calling simple_if_isinstance_idiom_attr_b(args, kwargs) (line 116)
simple_if_isinstance_idiom_attr_b_call_result_3866 = invoke(stypy.reporting.localization.Localization(__file__, 116, 0), simple_if_isinstance_idiom_attr_b_3861, *[Foo_call_result_3864], **kwargs_3865)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
