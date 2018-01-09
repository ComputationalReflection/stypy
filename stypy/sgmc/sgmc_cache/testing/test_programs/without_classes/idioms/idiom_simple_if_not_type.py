
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
9: def simple_if_not_base1(a):
10:     b = "hi"
11:     if not type(a) is int:
12:         r = a / 3
13:         r2 = a[0]
14:         b = 3
15:     r3 = a / 3
16:     r4 = b / 3
17: 
18: 
19: def simple_if_not_base2(a):
20:     b = "hi"
21:     if not type(a) is int:
22:         r = a / 3
23:         r2 = a[0]
24:         b = 3
25:     r3 = a / 3
26:     r4 = b / 3
27: 
28: 
29: def simple_if_not_base3(a):
30:     b = "hi"
31:     if not type(a) is int:
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
47: def simple_if_not_call_int(a):
48:     b = "hi"
49:     if not type(a + a) is int:
50:         r = a / 3
51:         r2 = a[0]
52:         b = 3
53:     r3 = a / 3
54:     r4 = b / 3
55: 
56: 
57: def simple_if_not_call_str(a):
58:     b = "hi"
59:     if not type(concat(a, a)) is int:
60:         r = a / 3
61:         r2 = a[0]
62:         b = 3
63:     r3 = a / 3
64:     r4 = b / 3
65: 
66: 
67: simple_if_not_base1(theInt)
68: simple_if_not_base2(theStr)
69: simple_if_not_base3(union)
70: 
71: simple_if_not_call_int(theInt)
72: simple_if_not_call_str(union)
73: 
74: 
75: def simple_if_not_idiom_variant(a):
76:     b = "hi"
77:     if not type(a) is type(3):
78:         r = a / 3
79:         r2 = a[0]
80:         b = 3
81:     r3 = a / 3
82:     r4 = b / 3
83: 
84: 
85: simple_if_not_idiom_variant(union)
86: 
87: 
88: def simple_if_not_not_idiom(a):
89:     b = "hi"
90:     if not type(a) is 3:
91:         r = a / 3
92:         r2 = a[0]
93:         b = 3
94:     r3 = a / 3
95:     r4 = b / 3
96: 
97: 
98: simple_if_not_not_idiom(union)
99: 
100: 
101: #
102: class Foo:
103:     def __init__(self):
104:         self.attr = 4
105:         self.strattr = "bar"
106: 
107: 
108: def simple_if_not_idiom_attr(a):
109:     b = "hi"
110:     if not type(a.attr) is int:
111:         r = a.attr / 3
112:         r2 = a.attr[0]
113:         b = 3
114:     r3 = a.attr / 3
115:     r4 = b / 3
116: 
117: 
118: def simple_if_not_diom_attr_b(a):
119:     b = "hi"
120:     if not type(a.strattr) is int:
121:         r = a.attr / 3
122:         r2 = a.strattr[0]
123:         b = 3
124:     r3 = a.strattr / 3
125:     r4 = b / 3
126: 
127: 
128: simple_if_not_idiom_attr(Foo())
129: simple_if_not_diom_attr_b(Foo())
130: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 1):
int_3969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 9), 'int')
# Assigning a type to the variable 'theInt' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'theInt', int_3969)

# Assigning a Str to a Name (line 2):
str_3970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 9), 'str', 'hi')
# Assigning a type to the variable 'theStr' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'theStr', str_3970)

# Getting the type of 'True' (line 3)
True_3971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 3), 'True')
# Testing the type of an if condition (line 3)
if_condition_3972 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 3, 0), True_3971)
# Assigning a type to the variable 'if_condition_3972' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'if_condition_3972', if_condition_3972)
# SSA begins for if statement (line 3)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Num to a Name (line 4):
int_3973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 12), 'int')
# Assigning a type to the variable 'union' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'union', int_3973)
# SSA branch for the else part of an if statement (line 3)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 6):
str_3974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'str', 'hi')
# Assigning a type to the variable 'union' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'union', str_3974)
# SSA join for if statement (line 3)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def simple_if_not_base1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_not_base1'
    module_type_store = module_type_store.open_function_context('simple_if_not_base1', 9, 0, False)
    
    # Passed parameters checking function
    simple_if_not_base1.stypy_localization = localization
    simple_if_not_base1.stypy_type_of_self = None
    simple_if_not_base1.stypy_type_store = module_type_store
    simple_if_not_base1.stypy_function_name = 'simple_if_not_base1'
    simple_if_not_base1.stypy_param_names_list = ['a']
    simple_if_not_base1.stypy_varargs_param_name = None
    simple_if_not_base1.stypy_kwargs_param_name = None
    simple_if_not_base1.stypy_call_defaults = defaults
    simple_if_not_base1.stypy_call_varargs = varargs
    simple_if_not_base1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_not_base1', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_not_base1', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_not_base1(...)' code ##################

    
    # Assigning a Str to a Name (line 10):
    str_3975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'b', str_3975)
    
    # Type idiom detected: calculating its left and rigth part (line 11)
    # Getting the type of 'a' (line 11)
    a_3976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'a')
    # Getting the type of 'int' (line 11)
    int_3977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 22), 'int')
    
    (may_be_3978, more_types_in_union_3979) = may_not_be_type(a_3976, int_3977)

    if may_be_3978:

        if more_types_in_union_3979:
            # Runtime conditional SSA (line 11)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 11)
        a_3980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'a')
        # Assigning a type to the variable 'a' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'a', remove_type_from_union(a_3980, int_3977))
        
        # Assigning a BinOp to a Name (line 12):
        # Getting the type of 'a' (line 12)
        a_3981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'a')
        int_3982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 16), 'int')
        # Applying the binary operator 'div' (line 12)
        result_div_3983 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 12), 'div', a_3981, int_3982)
        
        # Assigning a type to the variable 'r' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'r', result_div_3983)
        
        # Assigning a Subscript to a Name (line 13):
        
        # Obtaining the type of the subscript
        int_3984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'int')
        # Getting the type of 'a' (line 13)
        a_3985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 13)
        getitem___3986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 13), a_3985, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 13)
        subscript_call_result_3987 = invoke(stypy.reporting.localization.Localization(__file__, 13, 13), getitem___3986, int_3984)
        
        # Assigning a type to the variable 'r2' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'r2', subscript_call_result_3987)
        
        # Assigning a Num to a Name (line 14):
        int_3988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'int')
        # Assigning a type to the variable 'b' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'b', int_3988)

        if more_types_in_union_3979:
            # SSA join for if statement (line 11)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 15):
    # Getting the type of 'a' (line 15)
    a_3989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 9), 'a')
    int_3990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 13), 'int')
    # Applying the binary operator 'div' (line 15)
    result_div_3991 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 9), 'div', a_3989, int_3990)
    
    # Assigning a type to the variable 'r3' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'r3', result_div_3991)
    
    # Assigning a BinOp to a Name (line 16):
    # Getting the type of 'b' (line 16)
    b_3992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 9), 'b')
    int_3993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'int')
    # Applying the binary operator 'div' (line 16)
    result_div_3994 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 9), 'div', b_3992, int_3993)
    
    # Assigning a type to the variable 'r4' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'r4', result_div_3994)
    
    # ################# End of 'simple_if_not_base1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_not_base1' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_3995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3995)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_not_base1'
    return stypy_return_type_3995

# Assigning a type to the variable 'simple_if_not_base1' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'simple_if_not_base1', simple_if_not_base1)

@norecursion
def simple_if_not_base2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_not_base2'
    module_type_store = module_type_store.open_function_context('simple_if_not_base2', 19, 0, False)
    
    # Passed parameters checking function
    simple_if_not_base2.stypy_localization = localization
    simple_if_not_base2.stypy_type_of_self = None
    simple_if_not_base2.stypy_type_store = module_type_store
    simple_if_not_base2.stypy_function_name = 'simple_if_not_base2'
    simple_if_not_base2.stypy_param_names_list = ['a']
    simple_if_not_base2.stypy_varargs_param_name = None
    simple_if_not_base2.stypy_kwargs_param_name = None
    simple_if_not_base2.stypy_call_defaults = defaults
    simple_if_not_base2.stypy_call_varargs = varargs
    simple_if_not_base2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_not_base2', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_not_base2', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_not_base2(...)' code ##################

    
    # Assigning a Str to a Name (line 20):
    str_3996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'b', str_3996)
    
    # Type idiom detected: calculating its left and rigth part (line 21)
    # Getting the type of 'a' (line 21)
    a_3997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'a')
    # Getting the type of 'int' (line 21)
    int_3998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'int')
    
    (may_be_3999, more_types_in_union_4000) = may_not_be_type(a_3997, int_3998)

    if may_be_3999:

        if more_types_in_union_4000:
            # Runtime conditional SSA (line 21)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 21)
        a_4001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'a')
        # Assigning a type to the variable 'a' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'a', remove_type_from_union(a_4001, int_3998))
        
        # Assigning a BinOp to a Name (line 22):
        # Getting the type of 'a' (line 22)
        a_4002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'a')
        int_4003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'int')
        # Applying the binary operator 'div' (line 22)
        result_div_4004 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 12), 'div', a_4002, int_4003)
        
        # Assigning a type to the variable 'r' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'r', result_div_4004)
        
        # Assigning a Subscript to a Name (line 23):
        
        # Obtaining the type of the subscript
        int_4005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'int')
        # Getting the type of 'a' (line 23)
        a_4006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 23)
        getitem___4007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 13), a_4006, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 23)
        subscript_call_result_4008 = invoke(stypy.reporting.localization.Localization(__file__, 23, 13), getitem___4007, int_4005)
        
        # Assigning a type to the variable 'r2' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'r2', subscript_call_result_4008)
        
        # Assigning a Num to a Name (line 24):
        int_4009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 12), 'int')
        # Assigning a type to the variable 'b' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'b', int_4009)

        if more_types_in_union_4000:
            # SSA join for if statement (line 21)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 25):
    # Getting the type of 'a' (line 25)
    a_4010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 9), 'a')
    int_4011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'int')
    # Applying the binary operator 'div' (line 25)
    result_div_4012 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 9), 'div', a_4010, int_4011)
    
    # Assigning a type to the variable 'r3' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'r3', result_div_4012)
    
    # Assigning a BinOp to a Name (line 26):
    # Getting the type of 'b' (line 26)
    b_4013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 9), 'b')
    int_4014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 13), 'int')
    # Applying the binary operator 'div' (line 26)
    result_div_4015 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 9), 'div', b_4013, int_4014)
    
    # Assigning a type to the variable 'r4' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'r4', result_div_4015)
    
    # ################# End of 'simple_if_not_base2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_not_base2' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_4016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4016)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_not_base2'
    return stypy_return_type_4016

# Assigning a type to the variable 'simple_if_not_base2' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'simple_if_not_base2', simple_if_not_base2)

@norecursion
def simple_if_not_base3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_not_base3'
    module_type_store = module_type_store.open_function_context('simple_if_not_base3', 29, 0, False)
    
    # Passed parameters checking function
    simple_if_not_base3.stypy_localization = localization
    simple_if_not_base3.stypy_type_of_self = None
    simple_if_not_base3.stypy_type_store = module_type_store
    simple_if_not_base3.stypy_function_name = 'simple_if_not_base3'
    simple_if_not_base3.stypy_param_names_list = ['a']
    simple_if_not_base3.stypy_varargs_param_name = None
    simple_if_not_base3.stypy_kwargs_param_name = None
    simple_if_not_base3.stypy_call_defaults = defaults
    simple_if_not_base3.stypy_call_varargs = varargs
    simple_if_not_base3.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_not_base3', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_not_base3', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_not_base3(...)' code ##################

    
    # Assigning a Str to a Name (line 30):
    str_4017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'b', str_4017)
    
    # Type idiom detected: calculating its left and rigth part (line 31)
    # Getting the type of 'a' (line 31)
    a_4018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'a')
    # Getting the type of 'int' (line 31)
    int_4019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 22), 'int')
    
    (may_be_4020, more_types_in_union_4021) = may_not_be_type(a_4018, int_4019)

    if may_be_4020:

        if more_types_in_union_4021:
            # Runtime conditional SSA (line 31)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 31)
        a_4022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'a')
        # Assigning a type to the variable 'a' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'a', remove_type_from_union(a_4022, int_4019))
        
        # Assigning a BinOp to a Name (line 32):
        # Getting the type of 'a' (line 32)
        a_4023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'a')
        int_4024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 16), 'int')
        # Applying the binary operator 'div' (line 32)
        result_div_4025 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 12), 'div', a_4023, int_4024)
        
        # Assigning a type to the variable 'r' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'r', result_div_4025)
        
        # Assigning a Subscript to a Name (line 33):
        
        # Obtaining the type of the subscript
        int_4026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 15), 'int')
        # Getting the type of 'a' (line 33)
        a_4027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 33)
        getitem___4028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 13), a_4027, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 33)
        subscript_call_result_4029 = invoke(stypy.reporting.localization.Localization(__file__, 33, 13), getitem___4028, int_4026)
        
        # Assigning a type to the variable 'r2' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'r2', subscript_call_result_4029)
        
        # Assigning a Num to a Name (line 34):
        int_4030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 12), 'int')
        # Assigning a type to the variable 'b' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'b', int_4030)

        if more_types_in_union_4021:
            # SSA join for if statement (line 31)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 35):
    # Getting the type of 'a' (line 35)
    a_4031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 9), 'a')
    int_4032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 13), 'int')
    # Applying the binary operator 'div' (line 35)
    result_div_4033 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 9), 'div', a_4031, int_4032)
    
    # Assigning a type to the variable 'r3' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'r3', result_div_4033)
    
    # Assigning a BinOp to a Name (line 36):
    # Getting the type of 'b' (line 36)
    b_4034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 9), 'b')
    int_4035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 13), 'int')
    # Applying the binary operator 'div' (line 36)
    result_div_4036 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 9), 'div', b_4034, int_4035)
    
    # Assigning a type to the variable 'r4' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'r4', result_div_4036)
    
    # ################# End of 'simple_if_not_base3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_not_base3' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_4037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4037)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_not_base3'
    return stypy_return_type_4037

# Assigning a type to the variable 'simple_if_not_base3' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'simple_if_not_base3', simple_if_not_base3)

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
    a_4038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'a')
    # Getting the type of 'b' (line 40)
    b_4039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'b')
    # Applying the binary operator '+' (line 40)
    result_add_4040 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 11), '+', a_4038, b_4039)
    
    # Assigning a type to the variable 'stypy_return_type' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type', result_add_4040)
    
    # ################# End of 'sum(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sum' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_4041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4041)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sum'
    return stypy_return_type_4041

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
    a_4043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'a', False)
    # Processing the call keyword arguments (line 44)
    kwargs_4044 = {}
    # Getting the type of 'str' (line 44)
    str_4042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'str', False)
    # Calling str(args, kwargs) (line 44)
    str_call_result_4045 = invoke(stypy.reporting.localization.Localization(__file__, 44, 11), str_4042, *[a_4043], **kwargs_4044)
    
    
    # Call to str(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'b' (line 44)
    b_4047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 24), 'b', False)
    # Processing the call keyword arguments (line 44)
    kwargs_4048 = {}
    # Getting the type of 'str' (line 44)
    str_4046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'str', False)
    # Calling str(args, kwargs) (line 44)
    str_call_result_4049 = invoke(stypy.reporting.localization.Localization(__file__, 44, 20), str_4046, *[b_4047], **kwargs_4048)
    
    # Applying the binary operator '+' (line 44)
    result_add_4050 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 11), '+', str_call_result_4045, str_call_result_4049)
    
    # Assigning a type to the variable 'stypy_return_type' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type', result_add_4050)
    
    # ################# End of 'concat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'concat' in the type store
    # Getting the type of 'stypy_return_type' (line 43)
    stypy_return_type_4051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4051)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'concat'
    return stypy_return_type_4051

# Assigning a type to the variable 'concat' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'concat', concat)

@norecursion
def simple_if_not_call_int(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_not_call_int'
    module_type_store = module_type_store.open_function_context('simple_if_not_call_int', 47, 0, False)
    
    # Passed parameters checking function
    simple_if_not_call_int.stypy_localization = localization
    simple_if_not_call_int.stypy_type_of_self = None
    simple_if_not_call_int.stypy_type_store = module_type_store
    simple_if_not_call_int.stypy_function_name = 'simple_if_not_call_int'
    simple_if_not_call_int.stypy_param_names_list = ['a']
    simple_if_not_call_int.stypy_varargs_param_name = None
    simple_if_not_call_int.stypy_kwargs_param_name = None
    simple_if_not_call_int.stypy_call_defaults = defaults
    simple_if_not_call_int.stypy_call_varargs = varargs
    simple_if_not_call_int.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_not_call_int', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_not_call_int', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_not_call_int(...)' code ##################

    
    # Assigning a Str to a Name (line 48):
    str_4052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'b', str_4052)
    
    # Type idiom detected: calculating its left and rigth part (line 49)
    # Getting the type of 'a' (line 49)
    a_4053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'a')
    # Getting the type of 'a' (line 49)
    a_4054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 20), 'a')
    # Applying the binary operator '+' (line 49)
    result_add_4055 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 16), '+', a_4053, a_4054)
    
    # Getting the type of 'int' (line 49)
    int_4056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 26), 'int')
    
    (may_be_4057, more_types_in_union_4058) = may_not_be_type(result_add_4055, int_4056)

    if may_be_4057:

        if more_types_in_union_4058:
            # Runtime conditional SSA (line 49)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 50):
        # Getting the type of 'a' (line 50)
        a_4059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'a')
        int_4060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 16), 'int')
        # Applying the binary operator 'div' (line 50)
        result_div_4061 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 12), 'div', a_4059, int_4060)
        
        # Assigning a type to the variable 'r' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'r', result_div_4061)
        
        # Assigning a Subscript to a Name (line 51):
        
        # Obtaining the type of the subscript
        int_4062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 15), 'int')
        # Getting the type of 'a' (line 51)
        a_4063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___4064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 13), a_4063, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_4065 = invoke(stypy.reporting.localization.Localization(__file__, 51, 13), getitem___4064, int_4062)
        
        # Assigning a type to the variable 'r2' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'r2', subscript_call_result_4065)
        
        # Assigning a Num to a Name (line 52):
        int_4066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 12), 'int')
        # Assigning a type to the variable 'b' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'b', int_4066)

        if more_types_in_union_4058:
            # SSA join for if statement (line 49)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 53):
    # Getting the type of 'a' (line 53)
    a_4067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 9), 'a')
    int_4068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 13), 'int')
    # Applying the binary operator 'div' (line 53)
    result_div_4069 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 9), 'div', a_4067, int_4068)
    
    # Assigning a type to the variable 'r3' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'r3', result_div_4069)
    
    # Assigning a BinOp to a Name (line 54):
    # Getting the type of 'b' (line 54)
    b_4070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 9), 'b')
    int_4071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 13), 'int')
    # Applying the binary operator 'div' (line 54)
    result_div_4072 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 9), 'div', b_4070, int_4071)
    
    # Assigning a type to the variable 'r4' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'r4', result_div_4072)
    
    # ################# End of 'simple_if_not_call_int(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_not_call_int' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_4073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4073)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_not_call_int'
    return stypy_return_type_4073

# Assigning a type to the variable 'simple_if_not_call_int' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'simple_if_not_call_int', simple_if_not_call_int)

@norecursion
def simple_if_not_call_str(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_not_call_str'
    module_type_store = module_type_store.open_function_context('simple_if_not_call_str', 57, 0, False)
    
    # Passed parameters checking function
    simple_if_not_call_str.stypy_localization = localization
    simple_if_not_call_str.stypy_type_of_self = None
    simple_if_not_call_str.stypy_type_store = module_type_store
    simple_if_not_call_str.stypy_function_name = 'simple_if_not_call_str'
    simple_if_not_call_str.stypy_param_names_list = ['a']
    simple_if_not_call_str.stypy_varargs_param_name = None
    simple_if_not_call_str.stypy_kwargs_param_name = None
    simple_if_not_call_str.stypy_call_defaults = defaults
    simple_if_not_call_str.stypy_call_varargs = varargs
    simple_if_not_call_str.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_not_call_str', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_not_call_str', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_not_call_str(...)' code ##################

    
    # Assigning a Str to a Name (line 58):
    str_4074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'b', str_4074)
    
    # Type idiom detected: calculating its left and rigth part (line 59)
    
    # Call to concat(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'a' (line 59)
    a_4076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 23), 'a', False)
    # Getting the type of 'a' (line 59)
    a_4077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 26), 'a', False)
    # Processing the call keyword arguments (line 59)
    kwargs_4078 = {}
    # Getting the type of 'concat' (line 59)
    concat_4075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'concat', False)
    # Calling concat(args, kwargs) (line 59)
    concat_call_result_4079 = invoke(stypy.reporting.localization.Localization(__file__, 59, 16), concat_4075, *[a_4076, a_4077], **kwargs_4078)
    
    # Getting the type of 'int' (line 59)
    int_4080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 33), 'int')
    
    (may_be_4081, more_types_in_union_4082) = may_not_be_type(concat_call_result_4079, int_4080)

    if may_be_4081:

        if more_types_in_union_4082:
            # Runtime conditional SSA (line 59)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 60):
        # Getting the type of 'a' (line 60)
        a_4083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'a')
        int_4084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 16), 'int')
        # Applying the binary operator 'div' (line 60)
        result_div_4085 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 12), 'div', a_4083, int_4084)
        
        # Assigning a type to the variable 'r' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'r', result_div_4085)
        
        # Assigning a Subscript to a Name (line 61):
        
        # Obtaining the type of the subscript
        int_4086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 15), 'int')
        # Getting the type of 'a' (line 61)
        a_4087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___4088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 13), a_4087, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_4089 = invoke(stypy.reporting.localization.Localization(__file__, 61, 13), getitem___4088, int_4086)
        
        # Assigning a type to the variable 'r2' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'r2', subscript_call_result_4089)
        
        # Assigning a Num to a Name (line 62):
        int_4090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 12), 'int')
        # Assigning a type to the variable 'b' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'b', int_4090)

        if more_types_in_union_4082:
            # SSA join for if statement (line 59)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 63):
    # Getting the type of 'a' (line 63)
    a_4091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 9), 'a')
    int_4092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 13), 'int')
    # Applying the binary operator 'div' (line 63)
    result_div_4093 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 9), 'div', a_4091, int_4092)
    
    # Assigning a type to the variable 'r3' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'r3', result_div_4093)
    
    # Assigning a BinOp to a Name (line 64):
    # Getting the type of 'b' (line 64)
    b_4094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 9), 'b')
    int_4095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 13), 'int')
    # Applying the binary operator 'div' (line 64)
    result_div_4096 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 9), 'div', b_4094, int_4095)
    
    # Assigning a type to the variable 'r4' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'r4', result_div_4096)
    
    # ################# End of 'simple_if_not_call_str(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_not_call_str' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_4097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4097)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_not_call_str'
    return stypy_return_type_4097

# Assigning a type to the variable 'simple_if_not_call_str' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'simple_if_not_call_str', simple_if_not_call_str)

# Call to simple_if_not_base1(...): (line 67)
# Processing the call arguments (line 67)
# Getting the type of 'theInt' (line 67)
theInt_4099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), 'theInt', False)
# Processing the call keyword arguments (line 67)
kwargs_4100 = {}
# Getting the type of 'simple_if_not_base1' (line 67)
simple_if_not_base1_4098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'simple_if_not_base1', False)
# Calling simple_if_not_base1(args, kwargs) (line 67)
simple_if_not_base1_call_result_4101 = invoke(stypy.reporting.localization.Localization(__file__, 67, 0), simple_if_not_base1_4098, *[theInt_4099], **kwargs_4100)


# Call to simple_if_not_base2(...): (line 68)
# Processing the call arguments (line 68)
# Getting the type of 'theStr' (line 68)
theStr_4103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 20), 'theStr', False)
# Processing the call keyword arguments (line 68)
kwargs_4104 = {}
# Getting the type of 'simple_if_not_base2' (line 68)
simple_if_not_base2_4102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'simple_if_not_base2', False)
# Calling simple_if_not_base2(args, kwargs) (line 68)
simple_if_not_base2_call_result_4105 = invoke(stypy.reporting.localization.Localization(__file__, 68, 0), simple_if_not_base2_4102, *[theStr_4103], **kwargs_4104)


# Call to simple_if_not_base3(...): (line 69)
# Processing the call arguments (line 69)
# Getting the type of 'union' (line 69)
union_4107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 20), 'union', False)
# Processing the call keyword arguments (line 69)
kwargs_4108 = {}
# Getting the type of 'simple_if_not_base3' (line 69)
simple_if_not_base3_4106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'simple_if_not_base3', False)
# Calling simple_if_not_base3(args, kwargs) (line 69)
simple_if_not_base3_call_result_4109 = invoke(stypy.reporting.localization.Localization(__file__, 69, 0), simple_if_not_base3_4106, *[union_4107], **kwargs_4108)


# Call to simple_if_not_call_int(...): (line 71)
# Processing the call arguments (line 71)
# Getting the type of 'theInt' (line 71)
theInt_4111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 23), 'theInt', False)
# Processing the call keyword arguments (line 71)
kwargs_4112 = {}
# Getting the type of 'simple_if_not_call_int' (line 71)
simple_if_not_call_int_4110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'simple_if_not_call_int', False)
# Calling simple_if_not_call_int(args, kwargs) (line 71)
simple_if_not_call_int_call_result_4113 = invoke(stypy.reporting.localization.Localization(__file__, 71, 0), simple_if_not_call_int_4110, *[theInt_4111], **kwargs_4112)


# Call to simple_if_not_call_str(...): (line 72)
# Processing the call arguments (line 72)
# Getting the type of 'union' (line 72)
union_4115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 'union', False)
# Processing the call keyword arguments (line 72)
kwargs_4116 = {}
# Getting the type of 'simple_if_not_call_str' (line 72)
simple_if_not_call_str_4114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'simple_if_not_call_str', False)
# Calling simple_if_not_call_str(args, kwargs) (line 72)
simple_if_not_call_str_call_result_4117 = invoke(stypy.reporting.localization.Localization(__file__, 72, 0), simple_if_not_call_str_4114, *[union_4115], **kwargs_4116)


@norecursion
def simple_if_not_idiom_variant(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_not_idiom_variant'
    module_type_store = module_type_store.open_function_context('simple_if_not_idiom_variant', 75, 0, False)
    
    # Passed parameters checking function
    simple_if_not_idiom_variant.stypy_localization = localization
    simple_if_not_idiom_variant.stypy_type_of_self = None
    simple_if_not_idiom_variant.stypy_type_store = module_type_store
    simple_if_not_idiom_variant.stypy_function_name = 'simple_if_not_idiom_variant'
    simple_if_not_idiom_variant.stypy_param_names_list = ['a']
    simple_if_not_idiom_variant.stypy_varargs_param_name = None
    simple_if_not_idiom_variant.stypy_kwargs_param_name = None
    simple_if_not_idiom_variant.stypy_call_defaults = defaults
    simple_if_not_idiom_variant.stypy_call_varargs = varargs
    simple_if_not_idiom_variant.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_not_idiom_variant', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_not_idiom_variant', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_not_idiom_variant(...)' code ##################

    
    # Assigning a Str to a Name (line 76):
    str_4118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'b', str_4118)
    
    # Type idiom detected: calculating its left and rigth part (line 77)
    # Getting the type of 'a' (line 77)
    a_4119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'a')
    
    # Call to type(...): (line 77)
    # Processing the call arguments (line 77)
    int_4121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 27), 'int')
    # Processing the call keyword arguments (line 77)
    kwargs_4122 = {}
    # Getting the type of 'type' (line 77)
    type_4120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 22), 'type', False)
    # Calling type(args, kwargs) (line 77)
    type_call_result_4123 = invoke(stypy.reporting.localization.Localization(__file__, 77, 22), type_4120, *[int_4121], **kwargs_4122)
    
    
    (may_be_4124, more_types_in_union_4125) = may_not_be_type(a_4119, type_call_result_4123)

    if may_be_4124:

        if more_types_in_union_4125:
            # Runtime conditional SSA (line 77)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 77)
        a_4126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'a')
        # Assigning a type to the variable 'a' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'a', remove_type_from_union(a_4126, type_call_result_4123))
        
        # Assigning a BinOp to a Name (line 78):
        # Getting the type of 'a' (line 78)
        a_4127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'a')
        int_4128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 16), 'int')
        # Applying the binary operator 'div' (line 78)
        result_div_4129 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 12), 'div', a_4127, int_4128)
        
        # Assigning a type to the variable 'r' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'r', result_div_4129)
        
        # Assigning a Subscript to a Name (line 79):
        
        # Obtaining the type of the subscript
        int_4130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 15), 'int')
        # Getting the type of 'a' (line 79)
        a_4131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 79)
        getitem___4132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 13), a_4131, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 79)
        subscript_call_result_4133 = invoke(stypy.reporting.localization.Localization(__file__, 79, 13), getitem___4132, int_4130)
        
        # Assigning a type to the variable 'r2' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'r2', subscript_call_result_4133)
        
        # Assigning a Num to a Name (line 80):
        int_4134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 12), 'int')
        # Assigning a type to the variable 'b' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'b', int_4134)

        if more_types_in_union_4125:
            # SSA join for if statement (line 77)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 81):
    # Getting the type of 'a' (line 81)
    a_4135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 9), 'a')
    int_4136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 13), 'int')
    # Applying the binary operator 'div' (line 81)
    result_div_4137 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 9), 'div', a_4135, int_4136)
    
    # Assigning a type to the variable 'r3' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'r3', result_div_4137)
    
    # Assigning a BinOp to a Name (line 82):
    # Getting the type of 'b' (line 82)
    b_4138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 9), 'b')
    int_4139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 13), 'int')
    # Applying the binary operator 'div' (line 82)
    result_div_4140 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 9), 'div', b_4138, int_4139)
    
    # Assigning a type to the variable 'r4' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'r4', result_div_4140)
    
    # ################# End of 'simple_if_not_idiom_variant(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_not_idiom_variant' in the type store
    # Getting the type of 'stypy_return_type' (line 75)
    stypy_return_type_4141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4141)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_not_idiom_variant'
    return stypy_return_type_4141

# Assigning a type to the variable 'simple_if_not_idiom_variant' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'simple_if_not_idiom_variant', simple_if_not_idiom_variant)

# Call to simple_if_not_idiom_variant(...): (line 85)
# Processing the call arguments (line 85)
# Getting the type of 'union' (line 85)
union_4143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 28), 'union', False)
# Processing the call keyword arguments (line 85)
kwargs_4144 = {}
# Getting the type of 'simple_if_not_idiom_variant' (line 85)
simple_if_not_idiom_variant_4142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'simple_if_not_idiom_variant', False)
# Calling simple_if_not_idiom_variant(args, kwargs) (line 85)
simple_if_not_idiom_variant_call_result_4145 = invoke(stypy.reporting.localization.Localization(__file__, 85, 0), simple_if_not_idiom_variant_4142, *[union_4143], **kwargs_4144)


@norecursion
def simple_if_not_not_idiom(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_not_not_idiom'
    module_type_store = module_type_store.open_function_context('simple_if_not_not_idiom', 88, 0, False)
    
    # Passed parameters checking function
    simple_if_not_not_idiom.stypy_localization = localization
    simple_if_not_not_idiom.stypy_type_of_self = None
    simple_if_not_not_idiom.stypy_type_store = module_type_store
    simple_if_not_not_idiom.stypy_function_name = 'simple_if_not_not_idiom'
    simple_if_not_not_idiom.stypy_param_names_list = ['a']
    simple_if_not_not_idiom.stypy_varargs_param_name = None
    simple_if_not_not_idiom.stypy_kwargs_param_name = None
    simple_if_not_not_idiom.stypy_call_defaults = defaults
    simple_if_not_not_idiom.stypy_call_varargs = varargs
    simple_if_not_not_idiom.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_not_not_idiom', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_not_not_idiom', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_not_not_idiom(...)' code ##################

    
    # Assigning a Str to a Name (line 89):
    str_4146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'b', str_4146)
    
    
    
    
    # Call to type(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'a' (line 90)
    a_4148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'a', False)
    # Processing the call keyword arguments (line 90)
    kwargs_4149 = {}
    # Getting the type of 'type' (line 90)
    type_4147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 11), 'type', False)
    # Calling type(args, kwargs) (line 90)
    type_call_result_4150 = invoke(stypy.reporting.localization.Localization(__file__, 90, 11), type_4147, *[a_4148], **kwargs_4149)
    
    int_4151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 22), 'int')
    # Applying the binary operator 'is' (line 90)
    result_is__4152 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 11), 'is', type_call_result_4150, int_4151)
    
    # Applying the 'not' unary operator (line 90)
    result_not__4153 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 7), 'not', result_is__4152)
    
    # Testing the type of an if condition (line 90)
    if_condition_4154 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 4), result_not__4153)
    # Assigning a type to the variable 'if_condition_4154' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'if_condition_4154', if_condition_4154)
    # SSA begins for if statement (line 90)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 91):
    # Getting the type of 'a' (line 91)
    a_4155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'a')
    int_4156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 16), 'int')
    # Applying the binary operator 'div' (line 91)
    result_div_4157 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 12), 'div', a_4155, int_4156)
    
    # Assigning a type to the variable 'r' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'r', result_div_4157)
    
    # Assigning a Subscript to a Name (line 92):
    
    # Obtaining the type of the subscript
    int_4158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 15), 'int')
    # Getting the type of 'a' (line 92)
    a_4159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 13), 'a')
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___4160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 13), a_4159, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_4161 = invoke(stypy.reporting.localization.Localization(__file__, 92, 13), getitem___4160, int_4158)
    
    # Assigning a type to the variable 'r2' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'r2', subscript_call_result_4161)
    
    # Assigning a Num to a Name (line 93):
    int_4162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 12), 'int')
    # Assigning a type to the variable 'b' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'b', int_4162)
    # SSA join for if statement (line 90)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 94):
    # Getting the type of 'a' (line 94)
    a_4163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 9), 'a')
    int_4164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 13), 'int')
    # Applying the binary operator 'div' (line 94)
    result_div_4165 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 9), 'div', a_4163, int_4164)
    
    # Assigning a type to the variable 'r3' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'r3', result_div_4165)
    
    # Assigning a BinOp to a Name (line 95):
    # Getting the type of 'b' (line 95)
    b_4166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 9), 'b')
    int_4167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 13), 'int')
    # Applying the binary operator 'div' (line 95)
    result_div_4168 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 9), 'div', b_4166, int_4167)
    
    # Assigning a type to the variable 'r4' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'r4', result_div_4168)
    
    # ################# End of 'simple_if_not_not_idiom(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_not_not_idiom' in the type store
    # Getting the type of 'stypy_return_type' (line 88)
    stypy_return_type_4169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4169)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_not_not_idiom'
    return stypy_return_type_4169

# Assigning a type to the variable 'simple_if_not_not_idiom' (line 88)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'simple_if_not_not_idiom', simple_if_not_not_idiom)

# Call to simple_if_not_not_idiom(...): (line 98)
# Processing the call arguments (line 98)
# Getting the type of 'union' (line 98)
union_4171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'union', False)
# Processing the call keyword arguments (line 98)
kwargs_4172 = {}
# Getting the type of 'simple_if_not_not_idiom' (line 98)
simple_if_not_not_idiom_4170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'simple_if_not_not_idiom', False)
# Calling simple_if_not_not_idiom(args, kwargs) (line 98)
simple_if_not_not_idiom_call_result_4173 = invoke(stypy.reporting.localization.Localization(__file__, 98, 0), simple_if_not_not_idiom_4170, *[union_4171], **kwargs_4172)

# Declaration of the 'Foo' class

class Foo:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 103, 4, False)
        # Assigning a type to the variable 'self' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'self', type_of_self)
        
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

        
        # Assigning a Num to a Attribute (line 104):
        int_4174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 20), 'int')
        # Getting the type of 'self' (line 104)
        self_4175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'self')
        # Setting the type of the member 'attr' of a type (line 104)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), self_4175, 'attr', int_4174)
        
        # Assigning a Str to a Attribute (line 105):
        str_4176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 23), 'str', 'bar')
        # Getting the type of 'self' (line 105)
        self_4177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'self')
        # Setting the type of the member 'strattr' of a type (line 105)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), self_4177, 'strattr', str_4176)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Foo' (line 102)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'Foo', Foo)

@norecursion
def simple_if_not_idiom_attr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_not_idiom_attr'
    module_type_store = module_type_store.open_function_context('simple_if_not_idiom_attr', 108, 0, False)
    
    # Passed parameters checking function
    simple_if_not_idiom_attr.stypy_localization = localization
    simple_if_not_idiom_attr.stypy_type_of_self = None
    simple_if_not_idiom_attr.stypy_type_store = module_type_store
    simple_if_not_idiom_attr.stypy_function_name = 'simple_if_not_idiom_attr'
    simple_if_not_idiom_attr.stypy_param_names_list = ['a']
    simple_if_not_idiom_attr.stypy_varargs_param_name = None
    simple_if_not_idiom_attr.stypy_kwargs_param_name = None
    simple_if_not_idiom_attr.stypy_call_defaults = defaults
    simple_if_not_idiom_attr.stypy_call_varargs = varargs
    simple_if_not_idiom_attr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_not_idiom_attr', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_not_idiom_attr', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_not_idiom_attr(...)' code ##################

    
    # Assigning a Str to a Name (line 109):
    str_4178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'b', str_4178)
    
    # Type idiom detected: calculating its left and rigth part (line 110)
    # Getting the type of 'a' (line 110)
    a_4179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'a')
    # Obtaining the member 'attr' of a type (line 110)
    attr_4180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 16), a_4179, 'attr')
    # Getting the type of 'int' (line 110)
    int_4181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 27), 'int')
    
    (may_be_4182, more_types_in_union_4183) = may_not_be_type(attr_4180, int_4181)

    if may_be_4182:

        if more_types_in_union_4183:
            # Runtime conditional SSA (line 110)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 110)
        a_4184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'a')
        # Obtaining the member 'attr' of a type (line 110)
        attr_4185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 4), a_4184, 'attr')
        # Setting the type of the member 'attr' of a type (line 110)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 4), a_4184, 'attr', remove_type_from_union(attr_4185, int_4181))
        
        # Assigning a BinOp to a Name (line 111):
        # Getting the type of 'a' (line 111)
        a_4186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'a')
        # Obtaining the member 'attr' of a type (line 111)
        attr_4187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), a_4186, 'attr')
        int_4188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 21), 'int')
        # Applying the binary operator 'div' (line 111)
        result_div_4189 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 12), 'div', attr_4187, int_4188)
        
        # Assigning a type to the variable 'r' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'r', result_div_4189)
        
        # Assigning a Subscript to a Name (line 112):
        
        # Obtaining the type of the subscript
        int_4190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 20), 'int')
        # Getting the type of 'a' (line 112)
        a_4191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 13), 'a')
        # Obtaining the member 'attr' of a type (line 112)
        attr_4192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 13), a_4191, 'attr')
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___4193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 13), attr_4192, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 112)
        subscript_call_result_4194 = invoke(stypy.reporting.localization.Localization(__file__, 112, 13), getitem___4193, int_4190)
        
        # Assigning a type to the variable 'r2' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'r2', subscript_call_result_4194)
        
        # Assigning a Num to a Name (line 113):
        int_4195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 12), 'int')
        # Assigning a type to the variable 'b' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'b', int_4195)

        if more_types_in_union_4183:
            # SSA join for if statement (line 110)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 114):
    # Getting the type of 'a' (line 114)
    a_4196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 9), 'a')
    # Obtaining the member 'attr' of a type (line 114)
    attr_4197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 9), a_4196, 'attr')
    int_4198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 18), 'int')
    # Applying the binary operator 'div' (line 114)
    result_div_4199 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 9), 'div', attr_4197, int_4198)
    
    # Assigning a type to the variable 'r3' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'r3', result_div_4199)
    
    # Assigning a BinOp to a Name (line 115):
    # Getting the type of 'b' (line 115)
    b_4200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 9), 'b')
    int_4201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 13), 'int')
    # Applying the binary operator 'div' (line 115)
    result_div_4202 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 9), 'div', b_4200, int_4201)
    
    # Assigning a type to the variable 'r4' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'r4', result_div_4202)
    
    # ################# End of 'simple_if_not_idiom_attr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_not_idiom_attr' in the type store
    # Getting the type of 'stypy_return_type' (line 108)
    stypy_return_type_4203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4203)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_not_idiom_attr'
    return stypy_return_type_4203

# Assigning a type to the variable 'simple_if_not_idiom_attr' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'simple_if_not_idiom_attr', simple_if_not_idiom_attr)

@norecursion
def simple_if_not_diom_attr_b(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_not_diom_attr_b'
    module_type_store = module_type_store.open_function_context('simple_if_not_diom_attr_b', 118, 0, False)
    
    # Passed parameters checking function
    simple_if_not_diom_attr_b.stypy_localization = localization
    simple_if_not_diom_attr_b.stypy_type_of_self = None
    simple_if_not_diom_attr_b.stypy_type_store = module_type_store
    simple_if_not_diom_attr_b.stypy_function_name = 'simple_if_not_diom_attr_b'
    simple_if_not_diom_attr_b.stypy_param_names_list = ['a']
    simple_if_not_diom_attr_b.stypy_varargs_param_name = None
    simple_if_not_diom_attr_b.stypy_kwargs_param_name = None
    simple_if_not_diom_attr_b.stypy_call_defaults = defaults
    simple_if_not_diom_attr_b.stypy_call_varargs = varargs
    simple_if_not_diom_attr_b.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_not_diom_attr_b', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_not_diom_attr_b', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_not_diom_attr_b(...)' code ##################

    
    # Assigning a Str to a Name (line 119):
    str_4204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'b', str_4204)
    
    # Type idiom detected: calculating its left and rigth part (line 120)
    # Getting the type of 'a' (line 120)
    a_4205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'a')
    # Obtaining the member 'strattr' of a type (line 120)
    strattr_4206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 16), a_4205, 'strattr')
    # Getting the type of 'int' (line 120)
    int_4207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 30), 'int')
    
    (may_be_4208, more_types_in_union_4209) = may_not_be_type(strattr_4206, int_4207)

    if may_be_4208:

        if more_types_in_union_4209:
            # Runtime conditional SSA (line 120)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 120)
        a_4210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'a')
        # Obtaining the member 'strattr' of a type (line 120)
        strattr_4211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 4), a_4210, 'strattr')
        # Setting the type of the member 'strattr' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 4), a_4210, 'strattr', remove_type_from_union(strattr_4211, int_4207))
        
        # Assigning a BinOp to a Name (line 121):
        # Getting the type of 'a' (line 121)
        a_4212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'a')
        # Obtaining the member 'attr' of a type (line 121)
        attr_4213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), a_4212, 'attr')
        int_4214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 21), 'int')
        # Applying the binary operator 'div' (line 121)
        result_div_4215 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 12), 'div', attr_4213, int_4214)
        
        # Assigning a type to the variable 'r' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'r', result_div_4215)
        
        # Assigning a Subscript to a Name (line 122):
        
        # Obtaining the type of the subscript
        int_4216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 23), 'int')
        # Getting the type of 'a' (line 122)
        a_4217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 13), 'a')
        # Obtaining the member 'strattr' of a type (line 122)
        strattr_4218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 13), a_4217, 'strattr')
        # Obtaining the member '__getitem__' of a type (line 122)
        getitem___4219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 13), strattr_4218, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
        subscript_call_result_4220 = invoke(stypy.reporting.localization.Localization(__file__, 122, 13), getitem___4219, int_4216)
        
        # Assigning a type to the variable 'r2' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'r2', subscript_call_result_4220)
        
        # Assigning a Num to a Name (line 123):
        int_4221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 12), 'int')
        # Assigning a type to the variable 'b' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'b', int_4221)

        if more_types_in_union_4209:
            # SSA join for if statement (line 120)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 124):
    # Getting the type of 'a' (line 124)
    a_4222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 9), 'a')
    # Obtaining the member 'strattr' of a type (line 124)
    strattr_4223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 9), a_4222, 'strattr')
    int_4224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 21), 'int')
    # Applying the binary operator 'div' (line 124)
    result_div_4225 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 9), 'div', strattr_4223, int_4224)
    
    # Assigning a type to the variable 'r3' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'r3', result_div_4225)
    
    # Assigning a BinOp to a Name (line 125):
    # Getting the type of 'b' (line 125)
    b_4226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 9), 'b')
    int_4227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 13), 'int')
    # Applying the binary operator 'div' (line 125)
    result_div_4228 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 9), 'div', b_4226, int_4227)
    
    # Assigning a type to the variable 'r4' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'r4', result_div_4228)
    
    # ################# End of 'simple_if_not_diom_attr_b(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_not_diom_attr_b' in the type store
    # Getting the type of 'stypy_return_type' (line 118)
    stypy_return_type_4229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4229)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_not_diom_attr_b'
    return stypy_return_type_4229

# Assigning a type to the variable 'simple_if_not_diom_attr_b' (line 118)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'simple_if_not_diom_attr_b', simple_if_not_diom_attr_b)

# Call to simple_if_not_idiom_attr(...): (line 128)
# Processing the call arguments (line 128)

# Call to Foo(...): (line 128)
# Processing the call keyword arguments (line 128)
kwargs_4232 = {}
# Getting the type of 'Foo' (line 128)
Foo_4231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 25), 'Foo', False)
# Calling Foo(args, kwargs) (line 128)
Foo_call_result_4233 = invoke(stypy.reporting.localization.Localization(__file__, 128, 25), Foo_4231, *[], **kwargs_4232)

# Processing the call keyword arguments (line 128)
kwargs_4234 = {}
# Getting the type of 'simple_if_not_idiom_attr' (line 128)
simple_if_not_idiom_attr_4230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), 'simple_if_not_idiom_attr', False)
# Calling simple_if_not_idiom_attr(args, kwargs) (line 128)
simple_if_not_idiom_attr_call_result_4235 = invoke(stypy.reporting.localization.Localization(__file__, 128, 0), simple_if_not_idiom_attr_4230, *[Foo_call_result_4233], **kwargs_4234)


# Call to simple_if_not_diom_attr_b(...): (line 129)
# Processing the call arguments (line 129)

# Call to Foo(...): (line 129)
# Processing the call keyword arguments (line 129)
kwargs_4238 = {}
# Getting the type of 'Foo' (line 129)
Foo_4237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 26), 'Foo', False)
# Calling Foo(args, kwargs) (line 129)
Foo_call_result_4239 = invoke(stypy.reporting.localization.Localization(__file__, 129, 26), Foo_4237, *[], **kwargs_4238)

# Processing the call keyword arguments (line 129)
kwargs_4240 = {}
# Getting the type of 'simple_if_not_diom_attr_b' (line 129)
simple_if_not_diom_attr_b_4236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 0), 'simple_if_not_diom_attr_b', False)
# Calling simple_if_not_diom_attr_b(args, kwargs) (line 129)
simple_if_not_diom_attr_b_call_result_4241 = invoke(stypy.reporting.localization.Localization(__file__, 129, 0), simple_if_not_diom_attr_b_4236, *[Foo_call_result_4239], **kwargs_4240)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
