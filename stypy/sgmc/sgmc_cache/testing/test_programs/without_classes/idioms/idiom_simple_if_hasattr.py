
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
9: def simple_if_hasattr_base1(a):
10:     b = "hi"
11:     if hasattr(a, '__div__'):
12:         r = a / 3
13:         r2 = a[0]
14:         b = 3
15:     r3 = a / 3
16:     r4 = b / 3
17: 
18: 
19: def simple_if_hasattr_base2(a):
20:     b = "hi"
21:     if hasattr(a, '__div__'):
22:         r = a / 3
23:         r2 = a[0]
24:         b = 3
25:     r3 = a / 3
26:     r4 = b / 3
27: 
28: 
29: def simple_if_hasattr_base3(a):
30:     b = "hi"
31:     if hasattr(a, '__div__'):
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
47: def simple_if_hasattr_call_int(a):
48:     b = "hi"
49:     if hasattr(sum(a, a), '__div__'):
50:         r = a / 3
51:         r2 = a[0]
52:         b = 3
53:     r3 = a / 3
54:     r4 = b / 3
55: 
56: 
57: def simple_if_hasattr_call_str(a):
58:     b = "hi"
59:     if not hasattr(concat(a, a), '__div__'):
60:         r = a / 3
61:         r2 = a[0]
62:         b = 3
63:     r3 = a / 3
64:     r4 = b / 3
65: 
66: 
67: simple_if_hasattr_base1(theInt)
68: simple_if_hasattr_base2(theStr)
69: simple_if_hasattr_base3(union)
70: 
71: simple_if_hasattr_call_int(theInt)
72: simple_if_hasattr_call_str(union)
73: 
74: 
75: def simple_if_not_hasattr_idiom(a):
76:     b = "hi"
77:     if not hasattr(a, '__div__'):
78:         r = a / 3
79:         r2 = a[0]
80:         b = 3
81:     r3 = a / 3
82:     r4 = b / 3
83: 
84: 
85: simple_if_not_hasattr_idiom(union)
86: 
87: 
88: #
89: class Foo:
90:     def __init__(self):
91:         self.attr = 4
92:         self.strattr = "bar"
93: 
94: 
95: def simple_if_hasattr_idiom_attr(a):
96:     b = "hi"
97:     if hasattr(a.attr, '__div__'):
98:         r = a.attr / 3
99:         r2 = a.attr[0]
100:         b = 3
101:     r3 = a.attr / 3
102:     r4 = b / 3
103: 
104: 
105: def simple_if_hasattr_idiom_attr_b(a):
106:     b = "hi"
107:     if not hasattr(a.strattr, '__div__'):
108:         r = a.attr / 3
109:         r2 = a.strattr[0]
110:         b = 3
111:     r3 = a.strattr / 3
112:     r4 = b / 3
113: 
114: 
115: simple_if_hasattr_idiom_attr(Foo())
116: simple_if_hasattr_idiom_attr_b(Foo())
117: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 1):
int_3285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 9), 'int')
# Assigning a type to the variable 'theInt' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'theInt', int_3285)

# Assigning a Str to a Name (line 2):
str_3286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 9), 'str', 'hi')
# Assigning a type to the variable 'theStr' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'theStr', str_3286)

# Getting the type of 'True' (line 3)
True_3287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 3), 'True')
# Testing the type of an if condition (line 3)
if_condition_3288 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 3, 0), True_3287)
# Assigning a type to the variable 'if_condition_3288' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'if_condition_3288', if_condition_3288)
# SSA begins for if statement (line 3)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Num to a Name (line 4):
int_3289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 12), 'int')
# Assigning a type to the variable 'union' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'union', int_3289)
# SSA branch for the else part of an if statement (line 3)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 6):
str_3290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'str', 'hi')
# Assigning a type to the variable 'union' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'union', str_3290)
# SSA join for if statement (line 3)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def simple_if_hasattr_base1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_hasattr_base1'
    module_type_store = module_type_store.open_function_context('simple_if_hasattr_base1', 9, 0, False)
    
    # Passed parameters checking function
    simple_if_hasattr_base1.stypy_localization = localization
    simple_if_hasattr_base1.stypy_type_of_self = None
    simple_if_hasattr_base1.stypy_type_store = module_type_store
    simple_if_hasattr_base1.stypy_function_name = 'simple_if_hasattr_base1'
    simple_if_hasattr_base1.stypy_param_names_list = ['a']
    simple_if_hasattr_base1.stypy_varargs_param_name = None
    simple_if_hasattr_base1.stypy_kwargs_param_name = None
    simple_if_hasattr_base1.stypy_call_defaults = defaults
    simple_if_hasattr_base1.stypy_call_varargs = varargs
    simple_if_hasattr_base1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_hasattr_base1', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_hasattr_base1', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_hasattr_base1(...)' code ##################

    
    # Assigning a Str to a Name (line 10):
    str_3291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'b', str_3291)
    
    # Type idiom detected: calculating its left and rigth part (line 11)
    str_3292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 18), 'str', '__div__')
    # Getting the type of 'a' (line 11)
    a_3293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 15), 'a')
    
    (may_be_3294, more_types_in_union_3295) = may_provide_member(str_3292, a_3293)

    if may_be_3294:

        if more_types_in_union_3295:
            # Runtime conditional SSA (line 11)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'a', remove_not_member_provider_from_union(a_3293, '__div__'))
        
        # Assigning a BinOp to a Name (line 12):
        # Getting the type of 'a' (line 12)
        a_3296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'a')
        int_3297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 16), 'int')
        # Applying the binary operator 'div' (line 12)
        result_div_3298 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 12), 'div', a_3296, int_3297)
        
        # Assigning a type to the variable 'r' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'r', result_div_3298)
        
        # Assigning a Subscript to a Name (line 13):
        
        # Obtaining the type of the subscript
        int_3299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'int')
        # Getting the type of 'a' (line 13)
        a_3300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 13)
        getitem___3301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 13), a_3300, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 13)
        subscript_call_result_3302 = invoke(stypy.reporting.localization.Localization(__file__, 13, 13), getitem___3301, int_3299)
        
        # Assigning a type to the variable 'r2' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'r2', subscript_call_result_3302)
        
        # Assigning a Num to a Name (line 14):
        int_3303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'int')
        # Assigning a type to the variable 'b' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'b', int_3303)

        if more_types_in_union_3295:
            # SSA join for if statement (line 11)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 15):
    # Getting the type of 'a' (line 15)
    a_3304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 9), 'a')
    int_3305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 13), 'int')
    # Applying the binary operator 'div' (line 15)
    result_div_3306 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 9), 'div', a_3304, int_3305)
    
    # Assigning a type to the variable 'r3' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'r3', result_div_3306)
    
    # Assigning a BinOp to a Name (line 16):
    # Getting the type of 'b' (line 16)
    b_3307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 9), 'b')
    int_3308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'int')
    # Applying the binary operator 'div' (line 16)
    result_div_3309 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 9), 'div', b_3307, int_3308)
    
    # Assigning a type to the variable 'r4' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'r4', result_div_3309)
    
    # ################# End of 'simple_if_hasattr_base1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_hasattr_base1' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_3310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3310)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_hasattr_base1'
    return stypy_return_type_3310

# Assigning a type to the variable 'simple_if_hasattr_base1' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'simple_if_hasattr_base1', simple_if_hasattr_base1)

@norecursion
def simple_if_hasattr_base2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_hasattr_base2'
    module_type_store = module_type_store.open_function_context('simple_if_hasattr_base2', 19, 0, False)
    
    # Passed parameters checking function
    simple_if_hasattr_base2.stypy_localization = localization
    simple_if_hasattr_base2.stypy_type_of_self = None
    simple_if_hasattr_base2.stypy_type_store = module_type_store
    simple_if_hasattr_base2.stypy_function_name = 'simple_if_hasattr_base2'
    simple_if_hasattr_base2.stypy_param_names_list = ['a']
    simple_if_hasattr_base2.stypy_varargs_param_name = None
    simple_if_hasattr_base2.stypy_kwargs_param_name = None
    simple_if_hasattr_base2.stypy_call_defaults = defaults
    simple_if_hasattr_base2.stypy_call_varargs = varargs
    simple_if_hasattr_base2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_hasattr_base2', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_hasattr_base2', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_hasattr_base2(...)' code ##################

    
    # Assigning a Str to a Name (line 20):
    str_3311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'b', str_3311)
    
    # Type idiom detected: calculating its left and rigth part (line 21)
    str_3312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 18), 'str', '__div__')
    # Getting the type of 'a' (line 21)
    a_3313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'a')
    
    (may_be_3314, more_types_in_union_3315) = may_provide_member(str_3312, a_3313)

    if may_be_3314:

        if more_types_in_union_3315:
            # Runtime conditional SSA (line 21)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'a', remove_not_member_provider_from_union(a_3313, '__div__'))
        
        # Assigning a BinOp to a Name (line 22):
        # Getting the type of 'a' (line 22)
        a_3316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'a')
        int_3317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'int')
        # Applying the binary operator 'div' (line 22)
        result_div_3318 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 12), 'div', a_3316, int_3317)
        
        # Assigning a type to the variable 'r' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'r', result_div_3318)
        
        # Assigning a Subscript to a Name (line 23):
        
        # Obtaining the type of the subscript
        int_3319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'int')
        # Getting the type of 'a' (line 23)
        a_3320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 23)
        getitem___3321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 13), a_3320, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 23)
        subscript_call_result_3322 = invoke(stypy.reporting.localization.Localization(__file__, 23, 13), getitem___3321, int_3319)
        
        # Assigning a type to the variable 'r2' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'r2', subscript_call_result_3322)
        
        # Assigning a Num to a Name (line 24):
        int_3323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 12), 'int')
        # Assigning a type to the variable 'b' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'b', int_3323)

        if more_types_in_union_3315:
            # SSA join for if statement (line 21)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 25):
    # Getting the type of 'a' (line 25)
    a_3324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 9), 'a')
    int_3325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'int')
    # Applying the binary operator 'div' (line 25)
    result_div_3326 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 9), 'div', a_3324, int_3325)
    
    # Assigning a type to the variable 'r3' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'r3', result_div_3326)
    
    # Assigning a BinOp to a Name (line 26):
    # Getting the type of 'b' (line 26)
    b_3327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 9), 'b')
    int_3328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 13), 'int')
    # Applying the binary operator 'div' (line 26)
    result_div_3329 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 9), 'div', b_3327, int_3328)
    
    # Assigning a type to the variable 'r4' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'r4', result_div_3329)
    
    # ################# End of 'simple_if_hasattr_base2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_hasattr_base2' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_3330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3330)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_hasattr_base2'
    return stypy_return_type_3330

# Assigning a type to the variable 'simple_if_hasattr_base2' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'simple_if_hasattr_base2', simple_if_hasattr_base2)

@norecursion
def simple_if_hasattr_base3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_hasattr_base3'
    module_type_store = module_type_store.open_function_context('simple_if_hasattr_base3', 29, 0, False)
    
    # Passed parameters checking function
    simple_if_hasattr_base3.stypy_localization = localization
    simple_if_hasattr_base3.stypy_type_of_self = None
    simple_if_hasattr_base3.stypy_type_store = module_type_store
    simple_if_hasattr_base3.stypy_function_name = 'simple_if_hasattr_base3'
    simple_if_hasattr_base3.stypy_param_names_list = ['a']
    simple_if_hasattr_base3.stypy_varargs_param_name = None
    simple_if_hasattr_base3.stypy_kwargs_param_name = None
    simple_if_hasattr_base3.stypy_call_defaults = defaults
    simple_if_hasattr_base3.stypy_call_varargs = varargs
    simple_if_hasattr_base3.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_hasattr_base3', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_hasattr_base3', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_hasattr_base3(...)' code ##################

    
    # Assigning a Str to a Name (line 30):
    str_3331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'b', str_3331)
    
    # Type idiom detected: calculating its left and rigth part (line 31)
    str_3332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 18), 'str', '__div__')
    # Getting the type of 'a' (line 31)
    a_3333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'a')
    
    (may_be_3334, more_types_in_union_3335) = may_provide_member(str_3332, a_3333)

    if may_be_3334:

        if more_types_in_union_3335:
            # Runtime conditional SSA (line 31)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'a', remove_not_member_provider_from_union(a_3333, '__div__'))
        
        # Assigning a BinOp to a Name (line 32):
        # Getting the type of 'a' (line 32)
        a_3336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'a')
        int_3337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 16), 'int')
        # Applying the binary operator 'div' (line 32)
        result_div_3338 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 12), 'div', a_3336, int_3337)
        
        # Assigning a type to the variable 'r' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'r', result_div_3338)
        
        # Assigning a Subscript to a Name (line 33):
        
        # Obtaining the type of the subscript
        int_3339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 15), 'int')
        # Getting the type of 'a' (line 33)
        a_3340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 33)
        getitem___3341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 13), a_3340, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 33)
        subscript_call_result_3342 = invoke(stypy.reporting.localization.Localization(__file__, 33, 13), getitem___3341, int_3339)
        
        # Assigning a type to the variable 'r2' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'r2', subscript_call_result_3342)
        
        # Assigning a Num to a Name (line 34):
        int_3343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 12), 'int')
        # Assigning a type to the variable 'b' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'b', int_3343)

        if more_types_in_union_3335:
            # SSA join for if statement (line 31)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 35):
    # Getting the type of 'a' (line 35)
    a_3344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 9), 'a')
    int_3345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 13), 'int')
    # Applying the binary operator 'div' (line 35)
    result_div_3346 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 9), 'div', a_3344, int_3345)
    
    # Assigning a type to the variable 'r3' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'r3', result_div_3346)
    
    # Assigning a BinOp to a Name (line 36):
    # Getting the type of 'b' (line 36)
    b_3347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 9), 'b')
    int_3348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 13), 'int')
    # Applying the binary operator 'div' (line 36)
    result_div_3349 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 9), 'div', b_3347, int_3348)
    
    # Assigning a type to the variable 'r4' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'r4', result_div_3349)
    
    # ################# End of 'simple_if_hasattr_base3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_hasattr_base3' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_3350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3350)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_hasattr_base3'
    return stypy_return_type_3350

# Assigning a type to the variable 'simple_if_hasattr_base3' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'simple_if_hasattr_base3', simple_if_hasattr_base3)

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
    a_3351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'a')
    # Getting the type of 'b' (line 40)
    b_3352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'b')
    # Applying the binary operator '+' (line 40)
    result_add_3353 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 11), '+', a_3351, b_3352)
    
    # Assigning a type to the variable 'stypy_return_type' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type', result_add_3353)
    
    # ################# End of 'sum(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sum' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_3354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3354)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sum'
    return stypy_return_type_3354

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
    a_3356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'a', False)
    # Processing the call keyword arguments (line 44)
    kwargs_3357 = {}
    # Getting the type of 'str' (line 44)
    str_3355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'str', False)
    # Calling str(args, kwargs) (line 44)
    str_call_result_3358 = invoke(stypy.reporting.localization.Localization(__file__, 44, 11), str_3355, *[a_3356], **kwargs_3357)
    
    
    # Call to str(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'b' (line 44)
    b_3360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 24), 'b', False)
    # Processing the call keyword arguments (line 44)
    kwargs_3361 = {}
    # Getting the type of 'str' (line 44)
    str_3359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'str', False)
    # Calling str(args, kwargs) (line 44)
    str_call_result_3362 = invoke(stypy.reporting.localization.Localization(__file__, 44, 20), str_3359, *[b_3360], **kwargs_3361)
    
    # Applying the binary operator '+' (line 44)
    result_add_3363 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 11), '+', str_call_result_3358, str_call_result_3362)
    
    # Assigning a type to the variable 'stypy_return_type' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type', result_add_3363)
    
    # ################# End of 'concat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'concat' in the type store
    # Getting the type of 'stypy_return_type' (line 43)
    stypy_return_type_3364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3364)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'concat'
    return stypy_return_type_3364

# Assigning a type to the variable 'concat' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'concat', concat)

@norecursion
def simple_if_hasattr_call_int(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_hasattr_call_int'
    module_type_store = module_type_store.open_function_context('simple_if_hasattr_call_int', 47, 0, False)
    
    # Passed parameters checking function
    simple_if_hasattr_call_int.stypy_localization = localization
    simple_if_hasattr_call_int.stypy_type_of_self = None
    simple_if_hasattr_call_int.stypy_type_store = module_type_store
    simple_if_hasattr_call_int.stypy_function_name = 'simple_if_hasattr_call_int'
    simple_if_hasattr_call_int.stypy_param_names_list = ['a']
    simple_if_hasattr_call_int.stypy_varargs_param_name = None
    simple_if_hasattr_call_int.stypy_kwargs_param_name = None
    simple_if_hasattr_call_int.stypy_call_defaults = defaults
    simple_if_hasattr_call_int.stypy_call_varargs = varargs
    simple_if_hasattr_call_int.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_hasattr_call_int', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_hasattr_call_int', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_hasattr_call_int(...)' code ##################

    
    # Assigning a Str to a Name (line 48):
    str_3365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'b', str_3365)
    
    # Type idiom detected: calculating its left and rigth part (line 49)
    str_3366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 26), 'str', '__div__')
    
    # Call to sum(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'a' (line 49)
    a_3368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'a', False)
    # Getting the type of 'a' (line 49)
    a_3369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 22), 'a', False)
    # Processing the call keyword arguments (line 49)
    kwargs_3370 = {}
    # Getting the type of 'sum' (line 49)
    sum_3367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'sum', False)
    # Calling sum(args, kwargs) (line 49)
    sum_call_result_3371 = invoke(stypy.reporting.localization.Localization(__file__, 49, 15), sum_3367, *[a_3368, a_3369], **kwargs_3370)
    
    
    (may_be_3372, more_types_in_union_3373) = may_provide_member(str_3366, sum_call_result_3371)

    if may_be_3372:

        if more_types_in_union_3373:
            # Runtime conditional SSA (line 49)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 50):
        # Getting the type of 'a' (line 50)
        a_3374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'a')
        int_3375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 16), 'int')
        # Applying the binary operator 'div' (line 50)
        result_div_3376 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 12), 'div', a_3374, int_3375)
        
        # Assigning a type to the variable 'r' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'r', result_div_3376)
        
        # Assigning a Subscript to a Name (line 51):
        
        # Obtaining the type of the subscript
        int_3377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 15), 'int')
        # Getting the type of 'a' (line 51)
        a_3378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___3379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 13), a_3378, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_3380 = invoke(stypy.reporting.localization.Localization(__file__, 51, 13), getitem___3379, int_3377)
        
        # Assigning a type to the variable 'r2' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'r2', subscript_call_result_3380)
        
        # Assigning a Num to a Name (line 52):
        int_3381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 12), 'int')
        # Assigning a type to the variable 'b' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'b', int_3381)

        if more_types_in_union_3373:
            # SSA join for if statement (line 49)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 53):
    # Getting the type of 'a' (line 53)
    a_3382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 9), 'a')
    int_3383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 13), 'int')
    # Applying the binary operator 'div' (line 53)
    result_div_3384 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 9), 'div', a_3382, int_3383)
    
    # Assigning a type to the variable 'r3' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'r3', result_div_3384)
    
    # Assigning a BinOp to a Name (line 54):
    # Getting the type of 'b' (line 54)
    b_3385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 9), 'b')
    int_3386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 13), 'int')
    # Applying the binary operator 'div' (line 54)
    result_div_3387 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 9), 'div', b_3385, int_3386)
    
    # Assigning a type to the variable 'r4' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'r4', result_div_3387)
    
    # ################# End of 'simple_if_hasattr_call_int(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_hasattr_call_int' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_3388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3388)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_hasattr_call_int'
    return stypy_return_type_3388

# Assigning a type to the variable 'simple_if_hasattr_call_int' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'simple_if_hasattr_call_int', simple_if_hasattr_call_int)

@norecursion
def simple_if_hasattr_call_str(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_hasattr_call_str'
    module_type_store = module_type_store.open_function_context('simple_if_hasattr_call_str', 57, 0, False)
    
    # Passed parameters checking function
    simple_if_hasattr_call_str.stypy_localization = localization
    simple_if_hasattr_call_str.stypy_type_of_self = None
    simple_if_hasattr_call_str.stypy_type_store = module_type_store
    simple_if_hasattr_call_str.stypy_function_name = 'simple_if_hasattr_call_str'
    simple_if_hasattr_call_str.stypy_param_names_list = ['a']
    simple_if_hasattr_call_str.stypy_varargs_param_name = None
    simple_if_hasattr_call_str.stypy_kwargs_param_name = None
    simple_if_hasattr_call_str.stypy_call_defaults = defaults
    simple_if_hasattr_call_str.stypy_call_varargs = varargs
    simple_if_hasattr_call_str.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_hasattr_call_str', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_hasattr_call_str', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_hasattr_call_str(...)' code ##################

    
    # Assigning a Str to a Name (line 58):
    str_3389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'b', str_3389)
    
    # Type idiom detected: calculating its left and rigth part (line 59)
    str_3390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 33), 'str', '__div__')
    
    # Call to concat(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'a' (line 59)
    a_3392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 26), 'a', False)
    # Getting the type of 'a' (line 59)
    a_3393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 29), 'a', False)
    # Processing the call keyword arguments (line 59)
    kwargs_3394 = {}
    # Getting the type of 'concat' (line 59)
    concat_3391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 19), 'concat', False)
    # Calling concat(args, kwargs) (line 59)
    concat_call_result_3395 = invoke(stypy.reporting.localization.Localization(__file__, 59, 19), concat_3391, *[a_3392, a_3393], **kwargs_3394)
    
    
    (may_be_3396, more_types_in_union_3397) = may_not_provide_member(str_3390, concat_call_result_3395)

    if may_be_3396:

        if more_types_in_union_3397:
            # Runtime conditional SSA (line 59)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 60):
        # Getting the type of 'a' (line 60)
        a_3398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'a')
        int_3399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 16), 'int')
        # Applying the binary operator 'div' (line 60)
        result_div_3400 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 12), 'div', a_3398, int_3399)
        
        # Assigning a type to the variable 'r' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'r', result_div_3400)
        
        # Assigning a Subscript to a Name (line 61):
        
        # Obtaining the type of the subscript
        int_3401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 15), 'int')
        # Getting the type of 'a' (line 61)
        a_3402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___3403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 13), a_3402, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_3404 = invoke(stypy.reporting.localization.Localization(__file__, 61, 13), getitem___3403, int_3401)
        
        # Assigning a type to the variable 'r2' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'r2', subscript_call_result_3404)
        
        # Assigning a Num to a Name (line 62):
        int_3405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 12), 'int')
        # Assigning a type to the variable 'b' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'b', int_3405)

        if more_types_in_union_3397:
            # SSA join for if statement (line 59)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 63):
    # Getting the type of 'a' (line 63)
    a_3406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 9), 'a')
    int_3407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 13), 'int')
    # Applying the binary operator 'div' (line 63)
    result_div_3408 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 9), 'div', a_3406, int_3407)
    
    # Assigning a type to the variable 'r3' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'r3', result_div_3408)
    
    # Assigning a BinOp to a Name (line 64):
    # Getting the type of 'b' (line 64)
    b_3409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 9), 'b')
    int_3410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 13), 'int')
    # Applying the binary operator 'div' (line 64)
    result_div_3411 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 9), 'div', b_3409, int_3410)
    
    # Assigning a type to the variable 'r4' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'r4', result_div_3411)
    
    # ################# End of 'simple_if_hasattr_call_str(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_hasattr_call_str' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_3412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3412)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_hasattr_call_str'
    return stypy_return_type_3412

# Assigning a type to the variable 'simple_if_hasattr_call_str' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'simple_if_hasattr_call_str', simple_if_hasattr_call_str)

# Call to simple_if_hasattr_base1(...): (line 67)
# Processing the call arguments (line 67)
# Getting the type of 'theInt' (line 67)
theInt_3414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'theInt', False)
# Processing the call keyword arguments (line 67)
kwargs_3415 = {}
# Getting the type of 'simple_if_hasattr_base1' (line 67)
simple_if_hasattr_base1_3413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'simple_if_hasattr_base1', False)
# Calling simple_if_hasattr_base1(args, kwargs) (line 67)
simple_if_hasattr_base1_call_result_3416 = invoke(stypy.reporting.localization.Localization(__file__, 67, 0), simple_if_hasattr_base1_3413, *[theInt_3414], **kwargs_3415)


# Call to simple_if_hasattr_base2(...): (line 68)
# Processing the call arguments (line 68)
# Getting the type of 'theStr' (line 68)
theStr_3418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 24), 'theStr', False)
# Processing the call keyword arguments (line 68)
kwargs_3419 = {}
# Getting the type of 'simple_if_hasattr_base2' (line 68)
simple_if_hasattr_base2_3417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'simple_if_hasattr_base2', False)
# Calling simple_if_hasattr_base2(args, kwargs) (line 68)
simple_if_hasattr_base2_call_result_3420 = invoke(stypy.reporting.localization.Localization(__file__, 68, 0), simple_if_hasattr_base2_3417, *[theStr_3418], **kwargs_3419)


# Call to simple_if_hasattr_base3(...): (line 69)
# Processing the call arguments (line 69)
# Getting the type of 'union' (line 69)
union_3422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 24), 'union', False)
# Processing the call keyword arguments (line 69)
kwargs_3423 = {}
# Getting the type of 'simple_if_hasattr_base3' (line 69)
simple_if_hasattr_base3_3421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'simple_if_hasattr_base3', False)
# Calling simple_if_hasattr_base3(args, kwargs) (line 69)
simple_if_hasattr_base3_call_result_3424 = invoke(stypy.reporting.localization.Localization(__file__, 69, 0), simple_if_hasattr_base3_3421, *[union_3422], **kwargs_3423)


# Call to simple_if_hasattr_call_int(...): (line 71)
# Processing the call arguments (line 71)
# Getting the type of 'theInt' (line 71)
theInt_3426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), 'theInt', False)
# Processing the call keyword arguments (line 71)
kwargs_3427 = {}
# Getting the type of 'simple_if_hasattr_call_int' (line 71)
simple_if_hasattr_call_int_3425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'simple_if_hasattr_call_int', False)
# Calling simple_if_hasattr_call_int(args, kwargs) (line 71)
simple_if_hasattr_call_int_call_result_3428 = invoke(stypy.reporting.localization.Localization(__file__, 71, 0), simple_if_hasattr_call_int_3425, *[theInt_3426], **kwargs_3427)


# Call to simple_if_hasattr_call_str(...): (line 72)
# Processing the call arguments (line 72)
# Getting the type of 'union' (line 72)
union_3430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 27), 'union', False)
# Processing the call keyword arguments (line 72)
kwargs_3431 = {}
# Getting the type of 'simple_if_hasattr_call_str' (line 72)
simple_if_hasattr_call_str_3429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'simple_if_hasattr_call_str', False)
# Calling simple_if_hasattr_call_str(args, kwargs) (line 72)
simple_if_hasattr_call_str_call_result_3432 = invoke(stypy.reporting.localization.Localization(__file__, 72, 0), simple_if_hasattr_call_str_3429, *[union_3430], **kwargs_3431)


@norecursion
def simple_if_not_hasattr_idiom(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_not_hasattr_idiom'
    module_type_store = module_type_store.open_function_context('simple_if_not_hasattr_idiom', 75, 0, False)
    
    # Passed parameters checking function
    simple_if_not_hasattr_idiom.stypy_localization = localization
    simple_if_not_hasattr_idiom.stypy_type_of_self = None
    simple_if_not_hasattr_idiom.stypy_type_store = module_type_store
    simple_if_not_hasattr_idiom.stypy_function_name = 'simple_if_not_hasattr_idiom'
    simple_if_not_hasattr_idiom.stypy_param_names_list = ['a']
    simple_if_not_hasattr_idiom.stypy_varargs_param_name = None
    simple_if_not_hasattr_idiom.stypy_kwargs_param_name = None
    simple_if_not_hasattr_idiom.stypy_call_defaults = defaults
    simple_if_not_hasattr_idiom.stypy_call_varargs = varargs
    simple_if_not_hasattr_idiom.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_not_hasattr_idiom', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_not_hasattr_idiom', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_not_hasattr_idiom(...)' code ##################

    
    # Assigning a Str to a Name (line 76):
    str_3433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'b', str_3433)
    
    # Type idiom detected: calculating its left and rigth part (line 77)
    str_3434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 22), 'str', '__div__')
    # Getting the type of 'a' (line 77)
    a_3435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'a')
    
    (may_be_3436, more_types_in_union_3437) = may_not_provide_member(str_3434, a_3435)

    if may_be_3436:

        if more_types_in_union_3437:
            # Runtime conditional SSA (line 77)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'a', remove_member_provider_from_union(a_3435, '__div__'))
        
        # Assigning a BinOp to a Name (line 78):
        # Getting the type of 'a' (line 78)
        a_3438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'a')
        int_3439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 16), 'int')
        # Applying the binary operator 'div' (line 78)
        result_div_3440 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 12), 'div', a_3438, int_3439)
        
        # Assigning a type to the variable 'r' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'r', result_div_3440)
        
        # Assigning a Subscript to a Name (line 79):
        
        # Obtaining the type of the subscript
        int_3441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 15), 'int')
        # Getting the type of 'a' (line 79)
        a_3442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 79)
        getitem___3443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 13), a_3442, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 79)
        subscript_call_result_3444 = invoke(stypy.reporting.localization.Localization(__file__, 79, 13), getitem___3443, int_3441)
        
        # Assigning a type to the variable 'r2' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'r2', subscript_call_result_3444)
        
        # Assigning a Num to a Name (line 80):
        int_3445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 12), 'int')
        # Assigning a type to the variable 'b' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'b', int_3445)

        if more_types_in_union_3437:
            # SSA join for if statement (line 77)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 81):
    # Getting the type of 'a' (line 81)
    a_3446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 9), 'a')
    int_3447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 13), 'int')
    # Applying the binary operator 'div' (line 81)
    result_div_3448 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 9), 'div', a_3446, int_3447)
    
    # Assigning a type to the variable 'r3' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'r3', result_div_3448)
    
    # Assigning a BinOp to a Name (line 82):
    # Getting the type of 'b' (line 82)
    b_3449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 9), 'b')
    int_3450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 13), 'int')
    # Applying the binary operator 'div' (line 82)
    result_div_3451 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 9), 'div', b_3449, int_3450)
    
    # Assigning a type to the variable 'r4' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'r4', result_div_3451)
    
    # ################# End of 'simple_if_not_hasattr_idiom(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_not_hasattr_idiom' in the type store
    # Getting the type of 'stypy_return_type' (line 75)
    stypy_return_type_3452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3452)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_not_hasattr_idiom'
    return stypy_return_type_3452

# Assigning a type to the variable 'simple_if_not_hasattr_idiom' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'simple_if_not_hasattr_idiom', simple_if_not_hasattr_idiom)

# Call to simple_if_not_hasattr_idiom(...): (line 85)
# Processing the call arguments (line 85)
# Getting the type of 'union' (line 85)
union_3454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 28), 'union', False)
# Processing the call keyword arguments (line 85)
kwargs_3455 = {}
# Getting the type of 'simple_if_not_hasattr_idiom' (line 85)
simple_if_not_hasattr_idiom_3453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'simple_if_not_hasattr_idiom', False)
# Calling simple_if_not_hasattr_idiom(args, kwargs) (line 85)
simple_if_not_hasattr_idiom_call_result_3456 = invoke(stypy.reporting.localization.Localization(__file__, 85, 0), simple_if_not_hasattr_idiom_3453, *[union_3454], **kwargs_3455)

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
        int_3457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 20), 'int')
        # Getting the type of 'self' (line 91)
        self_3458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'self')
        # Setting the type of the member 'attr' of a type (line 91)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), self_3458, 'attr', int_3457)
        
        # Assigning a Str to a Attribute (line 92):
        str_3459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 23), 'str', 'bar')
        # Getting the type of 'self' (line 92)
        self_3460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'self')
        # Setting the type of the member 'strattr' of a type (line 92)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), self_3460, 'strattr', str_3459)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Foo' (line 89)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'Foo', Foo)

@norecursion
def simple_if_hasattr_idiom_attr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_hasattr_idiom_attr'
    module_type_store = module_type_store.open_function_context('simple_if_hasattr_idiom_attr', 95, 0, False)
    
    # Passed parameters checking function
    simple_if_hasattr_idiom_attr.stypy_localization = localization
    simple_if_hasattr_idiom_attr.stypy_type_of_self = None
    simple_if_hasattr_idiom_attr.stypy_type_store = module_type_store
    simple_if_hasattr_idiom_attr.stypy_function_name = 'simple_if_hasattr_idiom_attr'
    simple_if_hasattr_idiom_attr.stypy_param_names_list = ['a']
    simple_if_hasattr_idiom_attr.stypy_varargs_param_name = None
    simple_if_hasattr_idiom_attr.stypy_kwargs_param_name = None
    simple_if_hasattr_idiom_attr.stypy_call_defaults = defaults
    simple_if_hasattr_idiom_attr.stypy_call_varargs = varargs
    simple_if_hasattr_idiom_attr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_hasattr_idiom_attr', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_hasattr_idiom_attr', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_hasattr_idiom_attr(...)' code ##################

    
    # Assigning a Str to a Name (line 96):
    str_3461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'b', str_3461)
    
    # Type idiom detected: calculating its left and rigth part (line 97)
    str_3462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 23), 'str', '__div__')
    # Getting the type of 'a' (line 97)
    a_3463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'a')
    # Obtaining the member 'attr' of a type (line 97)
    attr_3464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 15), a_3463, 'attr')
    
    (may_be_3465, more_types_in_union_3466) = may_provide_member(str_3462, attr_3464)

    if may_be_3465:

        if more_types_in_union_3466:
            # Runtime conditional SSA (line 97)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 97)
        a_3467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'a')
        # Obtaining the member 'attr' of a type (line 97)
        attr_3468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 4), a_3467, 'attr')
        # Setting the type of the member 'attr' of a type (line 97)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 4), a_3467, 'attr', remove_not_member_provider_from_union(attr_3464, '__div__'))
        
        # Assigning a BinOp to a Name (line 98):
        # Getting the type of 'a' (line 98)
        a_3469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'a')
        # Obtaining the member 'attr' of a type (line 98)
        attr_3470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), a_3469, 'attr')
        int_3471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 21), 'int')
        # Applying the binary operator 'div' (line 98)
        result_div_3472 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 12), 'div', attr_3470, int_3471)
        
        # Assigning a type to the variable 'r' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'r', result_div_3472)
        
        # Assigning a Subscript to a Name (line 99):
        
        # Obtaining the type of the subscript
        int_3473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 20), 'int')
        # Getting the type of 'a' (line 99)
        a_3474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 13), 'a')
        # Obtaining the member 'attr' of a type (line 99)
        attr_3475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 13), a_3474, 'attr')
        # Obtaining the member '__getitem__' of a type (line 99)
        getitem___3476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 13), attr_3475, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 99)
        subscript_call_result_3477 = invoke(stypy.reporting.localization.Localization(__file__, 99, 13), getitem___3476, int_3473)
        
        # Assigning a type to the variable 'r2' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'r2', subscript_call_result_3477)
        
        # Assigning a Num to a Name (line 100):
        int_3478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 12), 'int')
        # Assigning a type to the variable 'b' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'b', int_3478)

        if more_types_in_union_3466:
            # SSA join for if statement (line 97)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 101):
    # Getting the type of 'a' (line 101)
    a_3479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 9), 'a')
    # Obtaining the member 'attr' of a type (line 101)
    attr_3480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 9), a_3479, 'attr')
    int_3481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 18), 'int')
    # Applying the binary operator 'div' (line 101)
    result_div_3482 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 9), 'div', attr_3480, int_3481)
    
    # Assigning a type to the variable 'r3' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'r3', result_div_3482)
    
    # Assigning a BinOp to a Name (line 102):
    # Getting the type of 'b' (line 102)
    b_3483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 9), 'b')
    int_3484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 13), 'int')
    # Applying the binary operator 'div' (line 102)
    result_div_3485 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 9), 'div', b_3483, int_3484)
    
    # Assigning a type to the variable 'r4' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'r4', result_div_3485)
    
    # ################# End of 'simple_if_hasattr_idiom_attr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_hasattr_idiom_attr' in the type store
    # Getting the type of 'stypy_return_type' (line 95)
    stypy_return_type_3486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3486)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_hasattr_idiom_attr'
    return stypy_return_type_3486

# Assigning a type to the variable 'simple_if_hasattr_idiom_attr' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'simple_if_hasattr_idiom_attr', simple_if_hasattr_idiom_attr)

@norecursion
def simple_if_hasattr_idiom_attr_b(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_hasattr_idiom_attr_b'
    module_type_store = module_type_store.open_function_context('simple_if_hasattr_idiom_attr_b', 105, 0, False)
    
    # Passed parameters checking function
    simple_if_hasattr_idiom_attr_b.stypy_localization = localization
    simple_if_hasattr_idiom_attr_b.stypy_type_of_self = None
    simple_if_hasattr_idiom_attr_b.stypy_type_store = module_type_store
    simple_if_hasattr_idiom_attr_b.stypy_function_name = 'simple_if_hasattr_idiom_attr_b'
    simple_if_hasattr_idiom_attr_b.stypy_param_names_list = ['a']
    simple_if_hasattr_idiom_attr_b.stypy_varargs_param_name = None
    simple_if_hasattr_idiom_attr_b.stypy_kwargs_param_name = None
    simple_if_hasattr_idiom_attr_b.stypy_call_defaults = defaults
    simple_if_hasattr_idiom_attr_b.stypy_call_varargs = varargs
    simple_if_hasattr_idiom_attr_b.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_hasattr_idiom_attr_b', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_hasattr_idiom_attr_b', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_hasattr_idiom_attr_b(...)' code ##################

    
    # Assigning a Str to a Name (line 106):
    str_3487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'b', str_3487)
    
    # Type idiom detected: calculating its left and rigth part (line 107)
    str_3488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 30), 'str', '__div__')
    # Getting the type of 'a' (line 107)
    a_3489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'a')
    # Obtaining the member 'strattr' of a type (line 107)
    strattr_3490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 19), a_3489, 'strattr')
    
    (may_be_3491, more_types_in_union_3492) = may_not_provide_member(str_3488, strattr_3490)

    if may_be_3491:

        if more_types_in_union_3492:
            # Runtime conditional SSA (line 107)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 107)
        a_3493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'a')
        # Obtaining the member 'strattr' of a type (line 107)
        strattr_3494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 4), a_3493, 'strattr')
        # Setting the type of the member 'strattr' of a type (line 107)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 4), a_3493, 'strattr', remove_member_provider_from_union(strattr_3490, '__div__'))
        
        # Assigning a BinOp to a Name (line 108):
        # Getting the type of 'a' (line 108)
        a_3495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'a')
        # Obtaining the member 'attr' of a type (line 108)
        attr_3496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), a_3495, 'attr')
        int_3497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 21), 'int')
        # Applying the binary operator 'div' (line 108)
        result_div_3498 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 12), 'div', attr_3496, int_3497)
        
        # Assigning a type to the variable 'r' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'r', result_div_3498)
        
        # Assigning a Subscript to a Name (line 109):
        
        # Obtaining the type of the subscript
        int_3499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 23), 'int')
        # Getting the type of 'a' (line 109)
        a_3500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 13), 'a')
        # Obtaining the member 'strattr' of a type (line 109)
        strattr_3501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 13), a_3500, 'strattr')
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___3502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 13), strattr_3501, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_3503 = invoke(stypy.reporting.localization.Localization(__file__, 109, 13), getitem___3502, int_3499)
        
        # Assigning a type to the variable 'r2' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'r2', subscript_call_result_3503)
        
        # Assigning a Num to a Name (line 110):
        int_3504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 12), 'int')
        # Assigning a type to the variable 'b' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'b', int_3504)

        if more_types_in_union_3492:
            # SSA join for if statement (line 107)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 111):
    # Getting the type of 'a' (line 111)
    a_3505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 9), 'a')
    # Obtaining the member 'strattr' of a type (line 111)
    strattr_3506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 9), a_3505, 'strattr')
    int_3507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 21), 'int')
    # Applying the binary operator 'div' (line 111)
    result_div_3508 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 9), 'div', strattr_3506, int_3507)
    
    # Assigning a type to the variable 'r3' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'r3', result_div_3508)
    
    # Assigning a BinOp to a Name (line 112):
    # Getting the type of 'b' (line 112)
    b_3509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 9), 'b')
    int_3510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 13), 'int')
    # Applying the binary operator 'div' (line 112)
    result_div_3511 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 9), 'div', b_3509, int_3510)
    
    # Assigning a type to the variable 'r4' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'r4', result_div_3511)
    
    # ################# End of 'simple_if_hasattr_idiom_attr_b(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_hasattr_idiom_attr_b' in the type store
    # Getting the type of 'stypy_return_type' (line 105)
    stypy_return_type_3512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3512)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_hasattr_idiom_attr_b'
    return stypy_return_type_3512

# Assigning a type to the variable 'simple_if_hasattr_idiom_attr_b' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'simple_if_hasattr_idiom_attr_b', simple_if_hasattr_idiom_attr_b)

# Call to simple_if_hasattr_idiom_attr(...): (line 115)
# Processing the call arguments (line 115)

# Call to Foo(...): (line 115)
# Processing the call keyword arguments (line 115)
kwargs_3515 = {}
# Getting the type of 'Foo' (line 115)
Foo_3514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 29), 'Foo', False)
# Calling Foo(args, kwargs) (line 115)
Foo_call_result_3516 = invoke(stypy.reporting.localization.Localization(__file__, 115, 29), Foo_3514, *[], **kwargs_3515)

# Processing the call keyword arguments (line 115)
kwargs_3517 = {}
# Getting the type of 'simple_if_hasattr_idiom_attr' (line 115)
simple_if_hasattr_idiom_attr_3513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'simple_if_hasattr_idiom_attr', False)
# Calling simple_if_hasattr_idiom_attr(args, kwargs) (line 115)
simple_if_hasattr_idiom_attr_call_result_3518 = invoke(stypy.reporting.localization.Localization(__file__, 115, 0), simple_if_hasattr_idiom_attr_3513, *[Foo_call_result_3516], **kwargs_3517)


# Call to simple_if_hasattr_idiom_attr_b(...): (line 116)
# Processing the call arguments (line 116)

# Call to Foo(...): (line 116)
# Processing the call keyword arguments (line 116)
kwargs_3521 = {}
# Getting the type of 'Foo' (line 116)
Foo_3520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), 'Foo', False)
# Calling Foo(args, kwargs) (line 116)
Foo_call_result_3522 = invoke(stypy.reporting.localization.Localization(__file__, 116, 31), Foo_3520, *[], **kwargs_3521)

# Processing the call keyword arguments (line 116)
kwargs_3523 = {}
# Getting the type of 'simple_if_hasattr_idiom_attr_b' (line 116)
simple_if_hasattr_idiom_attr_b_3519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'simple_if_hasattr_idiom_attr_b', False)
# Calling simple_if_hasattr_idiom_attr_b(args, kwargs) (line 116)
simple_if_hasattr_idiom_attr_b_call_result_3524 = invoke(stypy.reporting.localization.Localization(__file__, 116, 0), simple_if_hasattr_idiom_attr_b_3519, *[Foo_call_result_3522], **kwargs_3523)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
