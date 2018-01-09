
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
9: def simple_if_base1(a):
10:     b = "hi"
11:     if type(a) is int:
12:         r = a / 3
13:         r2 = a[0]
14:         b = 3
15:     r3 = a / 3
16:     r4 = b / 3
17: 
18: 
19: def simple_if_base2(a):
20:     b = "hi"
21:     if type(a) is int:
22:         r = a / 3
23:         r2 = a[0]
24:         b = 3
25:     r3 = a / 3
26:     r4 = b / 3
27: 
28: 
29: def simple_if_base3(a):
30:     b = "hi"
31:     if type(a) is int:
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
47: def simple_if_call_int(a):
48:     b = "hi"
49:     if type(a + a) is int:
50:         r = a / 3
51:         r2 = a[0]
52:         b = 3
53:     r3 = a / 3
54:     r4 = b / 3
55: 
56: 
57: def simple_if_call_str(a):
58:     b = "hi"
59:     if type(concat(a, a)) is int:
60:         r = a / 3
61:         r2 = a[0]
62:         b = 3
63:     r3 = a / 3
64:     r4 = b / 3
65: 
66: 
67: simple_if_base1(theInt)
68: simple_if_base2(theStr)
69: simple_if_base3(union)
70: 
71: simple_if_call_int(theInt)
72: simple_if_call_str(union)
73: 
74: 
75: def simple_if_idiom_variant(a):
76:     b = "hi"
77:     if type(a) is type(3):
78:         r = a / 3
79:         r2 = a[0]
80:         b = 3
81:     r3 = a / 3
82:     r4 = b / 3
83: 
84: 
85: simple_if_idiom_variant(union)
86: 
87: 
88: def simple_if_not_idiom(a):
89:     b = "hi"
90:     if type(a) is 3:
91:         r = a / 3
92:         r2 = a[0]
93:         b = 3
94:     r3 = a / 3
95:     r4 = b / 3
96: 
97: 
98: simple_if_not_idiom(union)
99: 
100: 
101: class Foo:
102:     def __init__(self):
103:         self.attr = 4
104:         self.strattr = "bar"
105: 
106: 
107: def simple_if_idiom_attr(a):
108:     b = "hi"
109:     if type(a.attr) is int:
110:         r = a.attr / 3
111:         r2 = a.attr[0]
112:         b = 3
113:     r3 = a.attr / 3
114:     r4 = b / 3
115: 
116: 
117: def simple_if_idiom_attr_b(a):
118:     b = "hi"
119:     if type(a.strattr) is int:
120:         r = a.attr / 3
121:         r2 = a.strattr[0]
122:         b = 3
123:     r3 = a.strattr / 3
124:     r4 = b / 3
125: 
126: 
127: simple_if_idiom_attr(Foo())
128: simple_if_idiom_attr_b(Foo())
129: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 1):
int_4448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 9), 'int')
# Assigning a type to the variable 'theInt' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'theInt', int_4448)

# Assigning a Str to a Name (line 2):
str_4449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 9), 'str', 'hi')
# Assigning a type to the variable 'theStr' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'theStr', str_4449)

# Getting the type of 'True' (line 3)
True_4450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 3), 'True')
# Testing the type of an if condition (line 3)
if_condition_4451 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 3, 0), True_4450)
# Assigning a type to the variable 'if_condition_4451' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'if_condition_4451', if_condition_4451)
# SSA begins for if statement (line 3)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Num to a Name (line 4):
int_4452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 12), 'int')
# Assigning a type to the variable 'union' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'union', int_4452)
# SSA branch for the else part of an if statement (line 3)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 6):
str_4453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'str', 'hi')
# Assigning a type to the variable 'union' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'union', str_4453)
# SSA join for if statement (line 3)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def simple_if_base1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_base1'
    module_type_store = module_type_store.open_function_context('simple_if_base1', 9, 0, False)
    
    # Passed parameters checking function
    simple_if_base1.stypy_localization = localization
    simple_if_base1.stypy_type_of_self = None
    simple_if_base1.stypy_type_store = module_type_store
    simple_if_base1.stypy_function_name = 'simple_if_base1'
    simple_if_base1.stypy_param_names_list = ['a']
    simple_if_base1.stypy_varargs_param_name = None
    simple_if_base1.stypy_kwargs_param_name = None
    simple_if_base1.stypy_call_defaults = defaults
    simple_if_base1.stypy_call_varargs = varargs
    simple_if_base1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_base1', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_base1', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_base1(...)' code ##################

    
    # Assigning a Str to a Name (line 10):
    str_4454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'b', str_4454)
    
    # Type idiom detected: calculating its left and rigth part (line 11)
    # Getting the type of 'a' (line 11)
    a_4455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'a')
    # Getting the type of 'int' (line 11)
    int_4456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 18), 'int')
    
    (may_be_4457, more_types_in_union_4458) = may_be_type(a_4455, int_4456)

    if may_be_4457:

        if more_types_in_union_4458:
            # Runtime conditional SSA (line 11)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'a', int_4456())
        
        # Assigning a BinOp to a Name (line 12):
        # Getting the type of 'a' (line 12)
        a_4459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'a')
        int_4460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 16), 'int')
        # Applying the binary operator 'div' (line 12)
        result_div_4461 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 12), 'div', a_4459, int_4460)
        
        # Assigning a type to the variable 'r' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'r', result_div_4461)
        
        # Assigning a Subscript to a Name (line 13):
        
        # Obtaining the type of the subscript
        int_4462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'int')
        # Getting the type of 'a' (line 13)
        a_4463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 13)
        getitem___4464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 13), a_4463, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 13)
        subscript_call_result_4465 = invoke(stypy.reporting.localization.Localization(__file__, 13, 13), getitem___4464, int_4462)
        
        # Assigning a type to the variable 'r2' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'r2', subscript_call_result_4465)
        
        # Assigning a Num to a Name (line 14):
        int_4466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'int')
        # Assigning a type to the variable 'b' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'b', int_4466)

        if more_types_in_union_4458:
            # SSA join for if statement (line 11)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 15):
    # Getting the type of 'a' (line 15)
    a_4467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 9), 'a')
    int_4468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 13), 'int')
    # Applying the binary operator 'div' (line 15)
    result_div_4469 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 9), 'div', a_4467, int_4468)
    
    # Assigning a type to the variable 'r3' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'r3', result_div_4469)
    
    # Assigning a BinOp to a Name (line 16):
    # Getting the type of 'b' (line 16)
    b_4470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 9), 'b')
    int_4471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'int')
    # Applying the binary operator 'div' (line 16)
    result_div_4472 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 9), 'div', b_4470, int_4471)
    
    # Assigning a type to the variable 'r4' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'r4', result_div_4472)
    
    # ################# End of 'simple_if_base1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_base1' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_4473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4473)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_base1'
    return stypy_return_type_4473

# Assigning a type to the variable 'simple_if_base1' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'simple_if_base1', simple_if_base1)

@norecursion
def simple_if_base2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_base2'
    module_type_store = module_type_store.open_function_context('simple_if_base2', 19, 0, False)
    
    # Passed parameters checking function
    simple_if_base2.stypy_localization = localization
    simple_if_base2.stypy_type_of_self = None
    simple_if_base2.stypy_type_store = module_type_store
    simple_if_base2.stypy_function_name = 'simple_if_base2'
    simple_if_base2.stypy_param_names_list = ['a']
    simple_if_base2.stypy_varargs_param_name = None
    simple_if_base2.stypy_kwargs_param_name = None
    simple_if_base2.stypy_call_defaults = defaults
    simple_if_base2.stypy_call_varargs = varargs
    simple_if_base2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_base2', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_base2', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_base2(...)' code ##################

    
    # Assigning a Str to a Name (line 20):
    str_4474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'b', str_4474)
    
    # Type idiom detected: calculating its left and rigth part (line 21)
    # Getting the type of 'a' (line 21)
    a_4475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'a')
    # Getting the type of 'int' (line 21)
    int_4476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 18), 'int')
    
    (may_be_4477, more_types_in_union_4478) = may_be_type(a_4475, int_4476)

    if may_be_4477:

        if more_types_in_union_4478:
            # Runtime conditional SSA (line 21)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'a', int_4476())
        
        # Assigning a BinOp to a Name (line 22):
        # Getting the type of 'a' (line 22)
        a_4479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'a')
        int_4480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'int')
        # Applying the binary operator 'div' (line 22)
        result_div_4481 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 12), 'div', a_4479, int_4480)
        
        # Assigning a type to the variable 'r' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'r', result_div_4481)
        
        # Assigning a Subscript to a Name (line 23):
        
        # Obtaining the type of the subscript
        int_4482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'int')
        # Getting the type of 'a' (line 23)
        a_4483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 23)
        getitem___4484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 13), a_4483, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 23)
        subscript_call_result_4485 = invoke(stypy.reporting.localization.Localization(__file__, 23, 13), getitem___4484, int_4482)
        
        # Assigning a type to the variable 'r2' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'r2', subscript_call_result_4485)
        
        # Assigning a Num to a Name (line 24):
        int_4486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 12), 'int')
        # Assigning a type to the variable 'b' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'b', int_4486)

        if more_types_in_union_4478:
            # SSA join for if statement (line 21)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 25):
    # Getting the type of 'a' (line 25)
    a_4487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 9), 'a')
    int_4488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'int')
    # Applying the binary operator 'div' (line 25)
    result_div_4489 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 9), 'div', a_4487, int_4488)
    
    # Assigning a type to the variable 'r3' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'r3', result_div_4489)
    
    # Assigning a BinOp to a Name (line 26):
    # Getting the type of 'b' (line 26)
    b_4490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 9), 'b')
    int_4491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 13), 'int')
    # Applying the binary operator 'div' (line 26)
    result_div_4492 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 9), 'div', b_4490, int_4491)
    
    # Assigning a type to the variable 'r4' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'r4', result_div_4492)
    
    # ################# End of 'simple_if_base2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_base2' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_4493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4493)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_base2'
    return stypy_return_type_4493

# Assigning a type to the variable 'simple_if_base2' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'simple_if_base2', simple_if_base2)

@norecursion
def simple_if_base3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_base3'
    module_type_store = module_type_store.open_function_context('simple_if_base3', 29, 0, False)
    
    # Passed parameters checking function
    simple_if_base3.stypy_localization = localization
    simple_if_base3.stypy_type_of_self = None
    simple_if_base3.stypy_type_store = module_type_store
    simple_if_base3.stypy_function_name = 'simple_if_base3'
    simple_if_base3.stypy_param_names_list = ['a']
    simple_if_base3.stypy_varargs_param_name = None
    simple_if_base3.stypy_kwargs_param_name = None
    simple_if_base3.stypy_call_defaults = defaults
    simple_if_base3.stypy_call_varargs = varargs
    simple_if_base3.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_base3', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_base3', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_base3(...)' code ##################

    
    # Assigning a Str to a Name (line 30):
    str_4494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'b', str_4494)
    
    # Type idiom detected: calculating its left and rigth part (line 31)
    # Getting the type of 'a' (line 31)
    a_4495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'a')
    # Getting the type of 'int' (line 31)
    int_4496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 18), 'int')
    
    (may_be_4497, more_types_in_union_4498) = may_be_type(a_4495, int_4496)

    if may_be_4497:

        if more_types_in_union_4498:
            # Runtime conditional SSA (line 31)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'a', int_4496())
        
        # Assigning a BinOp to a Name (line 32):
        # Getting the type of 'a' (line 32)
        a_4499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'a')
        int_4500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 16), 'int')
        # Applying the binary operator 'div' (line 32)
        result_div_4501 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 12), 'div', a_4499, int_4500)
        
        # Assigning a type to the variable 'r' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'r', result_div_4501)
        
        # Assigning a Subscript to a Name (line 33):
        
        # Obtaining the type of the subscript
        int_4502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 15), 'int')
        # Getting the type of 'a' (line 33)
        a_4503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 33)
        getitem___4504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 13), a_4503, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 33)
        subscript_call_result_4505 = invoke(stypy.reporting.localization.Localization(__file__, 33, 13), getitem___4504, int_4502)
        
        # Assigning a type to the variable 'r2' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'r2', subscript_call_result_4505)
        
        # Assigning a Num to a Name (line 34):
        int_4506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 12), 'int')
        # Assigning a type to the variable 'b' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'b', int_4506)

        if more_types_in_union_4498:
            # SSA join for if statement (line 31)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 35):
    # Getting the type of 'a' (line 35)
    a_4507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 9), 'a')
    int_4508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 13), 'int')
    # Applying the binary operator 'div' (line 35)
    result_div_4509 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 9), 'div', a_4507, int_4508)
    
    # Assigning a type to the variable 'r3' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'r3', result_div_4509)
    
    # Assigning a BinOp to a Name (line 36):
    # Getting the type of 'b' (line 36)
    b_4510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 9), 'b')
    int_4511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 13), 'int')
    # Applying the binary operator 'div' (line 36)
    result_div_4512 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 9), 'div', b_4510, int_4511)
    
    # Assigning a type to the variable 'r4' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'r4', result_div_4512)
    
    # ################# End of 'simple_if_base3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_base3' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_4513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4513)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_base3'
    return stypy_return_type_4513

# Assigning a type to the variable 'simple_if_base3' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'simple_if_base3', simple_if_base3)

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
    a_4514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'a')
    # Getting the type of 'b' (line 40)
    b_4515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'b')
    # Applying the binary operator '+' (line 40)
    result_add_4516 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 11), '+', a_4514, b_4515)
    
    # Assigning a type to the variable 'stypy_return_type' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type', result_add_4516)
    
    # ################# End of 'sum(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sum' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_4517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4517)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sum'
    return stypy_return_type_4517

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
    a_4519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'a', False)
    # Processing the call keyword arguments (line 44)
    kwargs_4520 = {}
    # Getting the type of 'str' (line 44)
    str_4518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'str', False)
    # Calling str(args, kwargs) (line 44)
    str_call_result_4521 = invoke(stypy.reporting.localization.Localization(__file__, 44, 11), str_4518, *[a_4519], **kwargs_4520)
    
    
    # Call to str(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'b' (line 44)
    b_4523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 24), 'b', False)
    # Processing the call keyword arguments (line 44)
    kwargs_4524 = {}
    # Getting the type of 'str' (line 44)
    str_4522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'str', False)
    # Calling str(args, kwargs) (line 44)
    str_call_result_4525 = invoke(stypy.reporting.localization.Localization(__file__, 44, 20), str_4522, *[b_4523], **kwargs_4524)
    
    # Applying the binary operator '+' (line 44)
    result_add_4526 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 11), '+', str_call_result_4521, str_call_result_4525)
    
    # Assigning a type to the variable 'stypy_return_type' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type', result_add_4526)
    
    # ################# End of 'concat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'concat' in the type store
    # Getting the type of 'stypy_return_type' (line 43)
    stypy_return_type_4527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4527)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'concat'
    return stypy_return_type_4527

# Assigning a type to the variable 'concat' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'concat', concat)

@norecursion
def simple_if_call_int(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_call_int'
    module_type_store = module_type_store.open_function_context('simple_if_call_int', 47, 0, False)
    
    # Passed parameters checking function
    simple_if_call_int.stypy_localization = localization
    simple_if_call_int.stypy_type_of_self = None
    simple_if_call_int.stypy_type_store = module_type_store
    simple_if_call_int.stypy_function_name = 'simple_if_call_int'
    simple_if_call_int.stypy_param_names_list = ['a']
    simple_if_call_int.stypy_varargs_param_name = None
    simple_if_call_int.stypy_kwargs_param_name = None
    simple_if_call_int.stypy_call_defaults = defaults
    simple_if_call_int.stypy_call_varargs = varargs
    simple_if_call_int.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_call_int', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_call_int', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_call_int(...)' code ##################

    
    # Assigning a Str to a Name (line 48):
    str_4528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'b', str_4528)
    
    # Type idiom detected: calculating its left and rigth part (line 49)
    # Getting the type of 'a' (line 49)
    a_4529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'a')
    # Getting the type of 'a' (line 49)
    a_4530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'a')
    # Applying the binary operator '+' (line 49)
    result_add_4531 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 12), '+', a_4529, a_4530)
    
    # Getting the type of 'int' (line 49)
    int_4532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 22), 'int')
    
    (may_be_4533, more_types_in_union_4534) = may_be_type(result_add_4531, int_4532)

    if may_be_4533:

        if more_types_in_union_4534:
            # Runtime conditional SSA (line 49)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 50):
        # Getting the type of 'a' (line 50)
        a_4535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'a')
        int_4536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 16), 'int')
        # Applying the binary operator 'div' (line 50)
        result_div_4537 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 12), 'div', a_4535, int_4536)
        
        # Assigning a type to the variable 'r' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'r', result_div_4537)
        
        # Assigning a Subscript to a Name (line 51):
        
        # Obtaining the type of the subscript
        int_4538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 15), 'int')
        # Getting the type of 'a' (line 51)
        a_4539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___4540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 13), a_4539, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_4541 = invoke(stypy.reporting.localization.Localization(__file__, 51, 13), getitem___4540, int_4538)
        
        # Assigning a type to the variable 'r2' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'r2', subscript_call_result_4541)
        
        # Assigning a Num to a Name (line 52):
        int_4542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 12), 'int')
        # Assigning a type to the variable 'b' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'b', int_4542)

        if more_types_in_union_4534:
            # SSA join for if statement (line 49)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 53):
    # Getting the type of 'a' (line 53)
    a_4543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 9), 'a')
    int_4544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 13), 'int')
    # Applying the binary operator 'div' (line 53)
    result_div_4545 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 9), 'div', a_4543, int_4544)
    
    # Assigning a type to the variable 'r3' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'r3', result_div_4545)
    
    # Assigning a BinOp to a Name (line 54):
    # Getting the type of 'b' (line 54)
    b_4546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 9), 'b')
    int_4547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 13), 'int')
    # Applying the binary operator 'div' (line 54)
    result_div_4548 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 9), 'div', b_4546, int_4547)
    
    # Assigning a type to the variable 'r4' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'r4', result_div_4548)
    
    # ################# End of 'simple_if_call_int(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_call_int' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_4549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4549)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_call_int'
    return stypy_return_type_4549

# Assigning a type to the variable 'simple_if_call_int' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'simple_if_call_int', simple_if_call_int)

@norecursion
def simple_if_call_str(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_call_str'
    module_type_store = module_type_store.open_function_context('simple_if_call_str', 57, 0, False)
    
    # Passed parameters checking function
    simple_if_call_str.stypy_localization = localization
    simple_if_call_str.stypy_type_of_self = None
    simple_if_call_str.stypy_type_store = module_type_store
    simple_if_call_str.stypy_function_name = 'simple_if_call_str'
    simple_if_call_str.stypy_param_names_list = ['a']
    simple_if_call_str.stypy_varargs_param_name = None
    simple_if_call_str.stypy_kwargs_param_name = None
    simple_if_call_str.stypy_call_defaults = defaults
    simple_if_call_str.stypy_call_varargs = varargs
    simple_if_call_str.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_call_str', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_call_str', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_call_str(...)' code ##################

    
    # Assigning a Str to a Name (line 58):
    str_4550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'b', str_4550)
    
    # Type idiom detected: calculating its left and rigth part (line 59)
    
    # Call to concat(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'a' (line 59)
    a_4552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 19), 'a', False)
    # Getting the type of 'a' (line 59)
    a_4553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 22), 'a', False)
    # Processing the call keyword arguments (line 59)
    kwargs_4554 = {}
    # Getting the type of 'concat' (line 59)
    concat_4551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'concat', False)
    # Calling concat(args, kwargs) (line 59)
    concat_call_result_4555 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), concat_4551, *[a_4552, a_4553], **kwargs_4554)
    
    # Getting the type of 'int' (line 59)
    int_4556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 29), 'int')
    
    (may_be_4557, more_types_in_union_4558) = may_be_type(concat_call_result_4555, int_4556)

    if may_be_4557:

        if more_types_in_union_4558:
            # Runtime conditional SSA (line 59)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 60):
        # Getting the type of 'a' (line 60)
        a_4559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'a')
        int_4560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 16), 'int')
        # Applying the binary operator 'div' (line 60)
        result_div_4561 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 12), 'div', a_4559, int_4560)
        
        # Assigning a type to the variable 'r' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'r', result_div_4561)
        
        # Assigning a Subscript to a Name (line 61):
        
        # Obtaining the type of the subscript
        int_4562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 15), 'int')
        # Getting the type of 'a' (line 61)
        a_4563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___4564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 13), a_4563, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_4565 = invoke(stypy.reporting.localization.Localization(__file__, 61, 13), getitem___4564, int_4562)
        
        # Assigning a type to the variable 'r2' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'r2', subscript_call_result_4565)
        
        # Assigning a Num to a Name (line 62):
        int_4566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 12), 'int')
        # Assigning a type to the variable 'b' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'b', int_4566)

        if more_types_in_union_4558:
            # SSA join for if statement (line 59)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 63):
    # Getting the type of 'a' (line 63)
    a_4567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 9), 'a')
    int_4568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 13), 'int')
    # Applying the binary operator 'div' (line 63)
    result_div_4569 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 9), 'div', a_4567, int_4568)
    
    # Assigning a type to the variable 'r3' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'r3', result_div_4569)
    
    # Assigning a BinOp to a Name (line 64):
    # Getting the type of 'b' (line 64)
    b_4570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 9), 'b')
    int_4571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 13), 'int')
    # Applying the binary operator 'div' (line 64)
    result_div_4572 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 9), 'div', b_4570, int_4571)
    
    # Assigning a type to the variable 'r4' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'r4', result_div_4572)
    
    # ################# End of 'simple_if_call_str(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_call_str' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_4573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4573)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_call_str'
    return stypy_return_type_4573

# Assigning a type to the variable 'simple_if_call_str' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'simple_if_call_str', simple_if_call_str)

# Call to simple_if_base1(...): (line 67)
# Processing the call arguments (line 67)
# Getting the type of 'theInt' (line 67)
theInt_4575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 16), 'theInt', False)
# Processing the call keyword arguments (line 67)
kwargs_4576 = {}
# Getting the type of 'simple_if_base1' (line 67)
simple_if_base1_4574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'simple_if_base1', False)
# Calling simple_if_base1(args, kwargs) (line 67)
simple_if_base1_call_result_4577 = invoke(stypy.reporting.localization.Localization(__file__, 67, 0), simple_if_base1_4574, *[theInt_4575], **kwargs_4576)


# Call to simple_if_base2(...): (line 68)
# Processing the call arguments (line 68)
# Getting the type of 'theStr' (line 68)
theStr_4579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'theStr', False)
# Processing the call keyword arguments (line 68)
kwargs_4580 = {}
# Getting the type of 'simple_if_base2' (line 68)
simple_if_base2_4578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'simple_if_base2', False)
# Calling simple_if_base2(args, kwargs) (line 68)
simple_if_base2_call_result_4581 = invoke(stypy.reporting.localization.Localization(__file__, 68, 0), simple_if_base2_4578, *[theStr_4579], **kwargs_4580)


# Call to simple_if_base3(...): (line 69)
# Processing the call arguments (line 69)
# Getting the type of 'union' (line 69)
union_4583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'union', False)
# Processing the call keyword arguments (line 69)
kwargs_4584 = {}
# Getting the type of 'simple_if_base3' (line 69)
simple_if_base3_4582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'simple_if_base3', False)
# Calling simple_if_base3(args, kwargs) (line 69)
simple_if_base3_call_result_4585 = invoke(stypy.reporting.localization.Localization(__file__, 69, 0), simple_if_base3_4582, *[union_4583], **kwargs_4584)


# Call to simple_if_call_int(...): (line 71)
# Processing the call arguments (line 71)
# Getting the type of 'theInt' (line 71)
theInt_4587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'theInt', False)
# Processing the call keyword arguments (line 71)
kwargs_4588 = {}
# Getting the type of 'simple_if_call_int' (line 71)
simple_if_call_int_4586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'simple_if_call_int', False)
# Calling simple_if_call_int(args, kwargs) (line 71)
simple_if_call_int_call_result_4589 = invoke(stypy.reporting.localization.Localization(__file__, 71, 0), simple_if_call_int_4586, *[theInt_4587], **kwargs_4588)


# Call to simple_if_call_str(...): (line 72)
# Processing the call arguments (line 72)
# Getting the type of 'union' (line 72)
union_4591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'union', False)
# Processing the call keyword arguments (line 72)
kwargs_4592 = {}
# Getting the type of 'simple_if_call_str' (line 72)
simple_if_call_str_4590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'simple_if_call_str', False)
# Calling simple_if_call_str(args, kwargs) (line 72)
simple_if_call_str_call_result_4593 = invoke(stypy.reporting.localization.Localization(__file__, 72, 0), simple_if_call_str_4590, *[union_4591], **kwargs_4592)


@norecursion
def simple_if_idiom_variant(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_idiom_variant'
    module_type_store = module_type_store.open_function_context('simple_if_idiom_variant', 75, 0, False)
    
    # Passed parameters checking function
    simple_if_idiom_variant.stypy_localization = localization
    simple_if_idiom_variant.stypy_type_of_self = None
    simple_if_idiom_variant.stypy_type_store = module_type_store
    simple_if_idiom_variant.stypy_function_name = 'simple_if_idiom_variant'
    simple_if_idiom_variant.stypy_param_names_list = ['a']
    simple_if_idiom_variant.stypy_varargs_param_name = None
    simple_if_idiom_variant.stypy_kwargs_param_name = None
    simple_if_idiom_variant.stypy_call_defaults = defaults
    simple_if_idiom_variant.stypy_call_varargs = varargs
    simple_if_idiom_variant.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_idiom_variant', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_idiom_variant', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_idiom_variant(...)' code ##################

    
    # Assigning a Str to a Name (line 76):
    str_4594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'b', str_4594)
    
    # Type idiom detected: calculating its left and rigth part (line 77)
    # Getting the type of 'a' (line 77)
    a_4595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'a')
    
    # Call to type(...): (line 77)
    # Processing the call arguments (line 77)
    int_4597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 23), 'int')
    # Processing the call keyword arguments (line 77)
    kwargs_4598 = {}
    # Getting the type of 'type' (line 77)
    type_4596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), 'type', False)
    # Calling type(args, kwargs) (line 77)
    type_call_result_4599 = invoke(stypy.reporting.localization.Localization(__file__, 77, 18), type_4596, *[int_4597], **kwargs_4598)
    
    
    (may_be_4600, more_types_in_union_4601) = may_be_type(a_4595, type_call_result_4599)

    if may_be_4600:

        if more_types_in_union_4601:
            # Runtime conditional SSA (line 77)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'a', type_call_result_4599())
        
        # Assigning a BinOp to a Name (line 78):
        # Getting the type of 'a' (line 78)
        a_4602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'a')
        int_4603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 16), 'int')
        # Applying the binary operator 'div' (line 78)
        result_div_4604 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 12), 'div', a_4602, int_4603)
        
        # Assigning a type to the variable 'r' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'r', result_div_4604)
        
        # Assigning a Subscript to a Name (line 79):
        
        # Obtaining the type of the subscript
        int_4605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 15), 'int')
        # Getting the type of 'a' (line 79)
        a_4606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 79)
        getitem___4607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 13), a_4606, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 79)
        subscript_call_result_4608 = invoke(stypy.reporting.localization.Localization(__file__, 79, 13), getitem___4607, int_4605)
        
        # Assigning a type to the variable 'r2' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'r2', subscript_call_result_4608)
        
        # Assigning a Num to a Name (line 80):
        int_4609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 12), 'int')
        # Assigning a type to the variable 'b' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'b', int_4609)

        if more_types_in_union_4601:
            # SSA join for if statement (line 77)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 81):
    # Getting the type of 'a' (line 81)
    a_4610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 9), 'a')
    int_4611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 13), 'int')
    # Applying the binary operator 'div' (line 81)
    result_div_4612 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 9), 'div', a_4610, int_4611)
    
    # Assigning a type to the variable 'r3' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'r3', result_div_4612)
    
    # Assigning a BinOp to a Name (line 82):
    # Getting the type of 'b' (line 82)
    b_4613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 9), 'b')
    int_4614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 13), 'int')
    # Applying the binary operator 'div' (line 82)
    result_div_4615 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 9), 'div', b_4613, int_4614)
    
    # Assigning a type to the variable 'r4' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'r4', result_div_4615)
    
    # ################# End of 'simple_if_idiom_variant(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_idiom_variant' in the type store
    # Getting the type of 'stypy_return_type' (line 75)
    stypy_return_type_4616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4616)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_idiom_variant'
    return stypy_return_type_4616

# Assigning a type to the variable 'simple_if_idiom_variant' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'simple_if_idiom_variant', simple_if_idiom_variant)

# Call to simple_if_idiom_variant(...): (line 85)
# Processing the call arguments (line 85)
# Getting the type of 'union' (line 85)
union_4618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 24), 'union', False)
# Processing the call keyword arguments (line 85)
kwargs_4619 = {}
# Getting the type of 'simple_if_idiom_variant' (line 85)
simple_if_idiom_variant_4617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'simple_if_idiom_variant', False)
# Calling simple_if_idiom_variant(args, kwargs) (line 85)
simple_if_idiom_variant_call_result_4620 = invoke(stypy.reporting.localization.Localization(__file__, 85, 0), simple_if_idiom_variant_4617, *[union_4618], **kwargs_4619)


@norecursion
def simple_if_not_idiom(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_not_idiom'
    module_type_store = module_type_store.open_function_context('simple_if_not_idiom', 88, 0, False)
    
    # Passed parameters checking function
    simple_if_not_idiom.stypy_localization = localization
    simple_if_not_idiom.stypy_type_of_self = None
    simple_if_not_idiom.stypy_type_store = module_type_store
    simple_if_not_idiom.stypy_function_name = 'simple_if_not_idiom'
    simple_if_not_idiom.stypy_param_names_list = ['a']
    simple_if_not_idiom.stypy_varargs_param_name = None
    simple_if_not_idiom.stypy_kwargs_param_name = None
    simple_if_not_idiom.stypy_call_defaults = defaults
    simple_if_not_idiom.stypy_call_varargs = varargs
    simple_if_not_idiom.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_not_idiom', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_not_idiom', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_not_idiom(...)' code ##################

    
    # Assigning a Str to a Name (line 89):
    str_4621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'b', str_4621)
    
    
    
    # Call to type(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'a' (line 90)
    a_4623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'a', False)
    # Processing the call keyword arguments (line 90)
    kwargs_4624 = {}
    # Getting the type of 'type' (line 90)
    type_4622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 7), 'type', False)
    # Calling type(args, kwargs) (line 90)
    type_call_result_4625 = invoke(stypy.reporting.localization.Localization(__file__, 90, 7), type_4622, *[a_4623], **kwargs_4624)
    
    int_4626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 18), 'int')
    # Applying the binary operator 'is' (line 90)
    result_is__4627 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 7), 'is', type_call_result_4625, int_4626)
    
    # Testing the type of an if condition (line 90)
    if_condition_4628 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 4), result_is__4627)
    # Assigning a type to the variable 'if_condition_4628' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'if_condition_4628', if_condition_4628)
    # SSA begins for if statement (line 90)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 91):
    # Getting the type of 'a' (line 91)
    a_4629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'a')
    int_4630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 16), 'int')
    # Applying the binary operator 'div' (line 91)
    result_div_4631 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 12), 'div', a_4629, int_4630)
    
    # Assigning a type to the variable 'r' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'r', result_div_4631)
    
    # Assigning a Subscript to a Name (line 92):
    
    # Obtaining the type of the subscript
    int_4632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 15), 'int')
    # Getting the type of 'a' (line 92)
    a_4633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 13), 'a')
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___4634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 13), a_4633, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_4635 = invoke(stypy.reporting.localization.Localization(__file__, 92, 13), getitem___4634, int_4632)
    
    # Assigning a type to the variable 'r2' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'r2', subscript_call_result_4635)
    
    # Assigning a Num to a Name (line 93):
    int_4636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 12), 'int')
    # Assigning a type to the variable 'b' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'b', int_4636)
    # SSA join for if statement (line 90)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 94):
    # Getting the type of 'a' (line 94)
    a_4637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 9), 'a')
    int_4638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 13), 'int')
    # Applying the binary operator 'div' (line 94)
    result_div_4639 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 9), 'div', a_4637, int_4638)
    
    # Assigning a type to the variable 'r3' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'r3', result_div_4639)
    
    # Assigning a BinOp to a Name (line 95):
    # Getting the type of 'b' (line 95)
    b_4640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 9), 'b')
    int_4641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 13), 'int')
    # Applying the binary operator 'div' (line 95)
    result_div_4642 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 9), 'div', b_4640, int_4641)
    
    # Assigning a type to the variable 'r4' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'r4', result_div_4642)
    
    # ################# End of 'simple_if_not_idiom(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_not_idiom' in the type store
    # Getting the type of 'stypy_return_type' (line 88)
    stypy_return_type_4643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4643)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_not_idiom'
    return stypy_return_type_4643

# Assigning a type to the variable 'simple_if_not_idiom' (line 88)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'simple_if_not_idiom', simple_if_not_idiom)

# Call to simple_if_not_idiom(...): (line 98)
# Processing the call arguments (line 98)
# Getting the type of 'union' (line 98)
union_4645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'union', False)
# Processing the call keyword arguments (line 98)
kwargs_4646 = {}
# Getting the type of 'simple_if_not_idiom' (line 98)
simple_if_not_idiom_4644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'simple_if_not_idiom', False)
# Calling simple_if_not_idiom(args, kwargs) (line 98)
simple_if_not_idiom_call_result_4647 = invoke(stypy.reporting.localization.Localization(__file__, 98, 0), simple_if_not_idiom_4644, *[union_4645], **kwargs_4646)

# Declaration of the 'Foo' class

class Foo:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 102, 4, False)
        # Assigning a type to the variable 'self' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'self', type_of_self)
        
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

        
        # Assigning a Num to a Attribute (line 103):
        int_4648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 20), 'int')
        # Getting the type of 'self' (line 103)
        self_4649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'self')
        # Setting the type of the member 'attr' of a type (line 103)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), self_4649, 'attr', int_4648)
        
        # Assigning a Str to a Attribute (line 104):
        str_4650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 23), 'str', 'bar')
        # Getting the type of 'self' (line 104)
        self_4651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'self')
        # Setting the type of the member 'strattr' of a type (line 104)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), self_4651, 'strattr', str_4650)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Foo' (line 101)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'Foo', Foo)

@norecursion
def simple_if_idiom_attr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_idiom_attr'
    module_type_store = module_type_store.open_function_context('simple_if_idiom_attr', 107, 0, False)
    
    # Passed parameters checking function
    simple_if_idiom_attr.stypy_localization = localization
    simple_if_idiom_attr.stypy_type_of_self = None
    simple_if_idiom_attr.stypy_type_store = module_type_store
    simple_if_idiom_attr.stypy_function_name = 'simple_if_idiom_attr'
    simple_if_idiom_attr.stypy_param_names_list = ['a']
    simple_if_idiom_attr.stypy_varargs_param_name = None
    simple_if_idiom_attr.stypy_kwargs_param_name = None
    simple_if_idiom_attr.stypy_call_defaults = defaults
    simple_if_idiom_attr.stypy_call_varargs = varargs
    simple_if_idiom_attr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_idiom_attr', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_idiom_attr', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_idiom_attr(...)' code ##################

    
    # Assigning a Str to a Name (line 108):
    str_4652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'b', str_4652)
    
    # Type idiom detected: calculating its left and rigth part (line 109)
    # Getting the type of 'a' (line 109)
    a_4653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'a')
    # Obtaining the member 'attr' of a type (line 109)
    attr_4654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 12), a_4653, 'attr')
    # Getting the type of 'int' (line 109)
    int_4655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 23), 'int')
    
    (may_be_4656, more_types_in_union_4657) = may_be_type(attr_4654, int_4655)

    if may_be_4656:

        if more_types_in_union_4657:
            # Runtime conditional SSA (line 109)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 109)
        a_4658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'a')
        # Setting the type of the member 'attr' of a type (line 109)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 4), a_4658, 'attr', int_4655())
        
        # Assigning a BinOp to a Name (line 110):
        # Getting the type of 'a' (line 110)
        a_4659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'a')
        # Obtaining the member 'attr' of a type (line 110)
        attr_4660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), a_4659, 'attr')
        int_4661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 21), 'int')
        # Applying the binary operator 'div' (line 110)
        result_div_4662 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 12), 'div', attr_4660, int_4661)
        
        # Assigning a type to the variable 'r' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'r', result_div_4662)
        
        # Assigning a Subscript to a Name (line 111):
        
        # Obtaining the type of the subscript
        int_4663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 20), 'int')
        # Getting the type of 'a' (line 111)
        a_4664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 13), 'a')
        # Obtaining the member 'attr' of a type (line 111)
        attr_4665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 13), a_4664, 'attr')
        # Obtaining the member '__getitem__' of a type (line 111)
        getitem___4666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 13), attr_4665, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 111)
        subscript_call_result_4667 = invoke(stypy.reporting.localization.Localization(__file__, 111, 13), getitem___4666, int_4663)
        
        # Assigning a type to the variable 'r2' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'r2', subscript_call_result_4667)
        
        # Assigning a Num to a Name (line 112):
        int_4668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 12), 'int')
        # Assigning a type to the variable 'b' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'b', int_4668)

        if more_types_in_union_4657:
            # SSA join for if statement (line 109)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 113):
    # Getting the type of 'a' (line 113)
    a_4669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 9), 'a')
    # Obtaining the member 'attr' of a type (line 113)
    attr_4670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 9), a_4669, 'attr')
    int_4671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 18), 'int')
    # Applying the binary operator 'div' (line 113)
    result_div_4672 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 9), 'div', attr_4670, int_4671)
    
    # Assigning a type to the variable 'r3' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'r3', result_div_4672)
    
    # Assigning a BinOp to a Name (line 114):
    # Getting the type of 'b' (line 114)
    b_4673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 9), 'b')
    int_4674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 13), 'int')
    # Applying the binary operator 'div' (line 114)
    result_div_4675 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 9), 'div', b_4673, int_4674)
    
    # Assigning a type to the variable 'r4' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'r4', result_div_4675)
    
    # ################# End of 'simple_if_idiom_attr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_idiom_attr' in the type store
    # Getting the type of 'stypy_return_type' (line 107)
    stypy_return_type_4676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4676)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_idiom_attr'
    return stypy_return_type_4676

# Assigning a type to the variable 'simple_if_idiom_attr' (line 107)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'simple_if_idiom_attr', simple_if_idiom_attr)

@norecursion
def simple_if_idiom_attr_b(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_idiom_attr_b'
    module_type_store = module_type_store.open_function_context('simple_if_idiom_attr_b', 117, 0, False)
    
    # Passed parameters checking function
    simple_if_idiom_attr_b.stypy_localization = localization
    simple_if_idiom_attr_b.stypy_type_of_self = None
    simple_if_idiom_attr_b.stypy_type_store = module_type_store
    simple_if_idiom_attr_b.stypy_function_name = 'simple_if_idiom_attr_b'
    simple_if_idiom_attr_b.stypy_param_names_list = ['a']
    simple_if_idiom_attr_b.stypy_varargs_param_name = None
    simple_if_idiom_attr_b.stypy_kwargs_param_name = None
    simple_if_idiom_attr_b.stypy_call_defaults = defaults
    simple_if_idiom_attr_b.stypy_call_varargs = varargs
    simple_if_idiom_attr_b.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_idiom_attr_b', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_idiom_attr_b', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_idiom_attr_b(...)' code ##################

    
    # Assigning a Str to a Name (line 118):
    str_4677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'b', str_4677)
    
    # Type idiom detected: calculating its left and rigth part (line 119)
    # Getting the type of 'a' (line 119)
    a_4678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'a')
    # Obtaining the member 'strattr' of a type (line 119)
    strattr_4679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 12), a_4678, 'strattr')
    # Getting the type of 'int' (line 119)
    int_4680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 26), 'int')
    
    (may_be_4681, more_types_in_union_4682) = may_be_type(strattr_4679, int_4680)

    if may_be_4681:

        if more_types_in_union_4682:
            # Runtime conditional SSA (line 119)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 119)
        a_4683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'a')
        # Setting the type of the member 'strattr' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 4), a_4683, 'strattr', int_4680())
        
        # Assigning a BinOp to a Name (line 120):
        # Getting the type of 'a' (line 120)
        a_4684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'a')
        # Obtaining the member 'attr' of a type (line 120)
        attr_4685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), a_4684, 'attr')
        int_4686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 21), 'int')
        # Applying the binary operator 'div' (line 120)
        result_div_4687 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 12), 'div', attr_4685, int_4686)
        
        # Assigning a type to the variable 'r' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'r', result_div_4687)
        
        # Assigning a Subscript to a Name (line 121):
        
        # Obtaining the type of the subscript
        int_4688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 23), 'int')
        # Getting the type of 'a' (line 121)
        a_4689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 13), 'a')
        # Obtaining the member 'strattr' of a type (line 121)
        strattr_4690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 13), a_4689, 'strattr')
        # Obtaining the member '__getitem__' of a type (line 121)
        getitem___4691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 13), strattr_4690, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 121)
        subscript_call_result_4692 = invoke(stypy.reporting.localization.Localization(__file__, 121, 13), getitem___4691, int_4688)
        
        # Assigning a type to the variable 'r2' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'r2', subscript_call_result_4692)
        
        # Assigning a Num to a Name (line 122):
        int_4693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 12), 'int')
        # Assigning a type to the variable 'b' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'b', int_4693)

        if more_types_in_union_4682:
            # SSA join for if statement (line 119)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 123):
    # Getting the type of 'a' (line 123)
    a_4694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 9), 'a')
    # Obtaining the member 'strattr' of a type (line 123)
    strattr_4695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 9), a_4694, 'strattr')
    int_4696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 21), 'int')
    # Applying the binary operator 'div' (line 123)
    result_div_4697 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 9), 'div', strattr_4695, int_4696)
    
    # Assigning a type to the variable 'r3' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'r3', result_div_4697)
    
    # Assigning a BinOp to a Name (line 124):
    # Getting the type of 'b' (line 124)
    b_4698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 9), 'b')
    int_4699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 13), 'int')
    # Applying the binary operator 'div' (line 124)
    result_div_4700 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 9), 'div', b_4698, int_4699)
    
    # Assigning a type to the variable 'r4' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'r4', result_div_4700)
    
    # ################# End of 'simple_if_idiom_attr_b(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_idiom_attr_b' in the type store
    # Getting the type of 'stypy_return_type' (line 117)
    stypy_return_type_4701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4701)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_idiom_attr_b'
    return stypy_return_type_4701

# Assigning a type to the variable 'simple_if_idiom_attr_b' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'simple_if_idiom_attr_b', simple_if_idiom_attr_b)

# Call to simple_if_idiom_attr(...): (line 127)
# Processing the call arguments (line 127)

# Call to Foo(...): (line 127)
# Processing the call keyword arguments (line 127)
kwargs_4704 = {}
# Getting the type of 'Foo' (line 127)
Foo_4703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 21), 'Foo', False)
# Calling Foo(args, kwargs) (line 127)
Foo_call_result_4705 = invoke(stypy.reporting.localization.Localization(__file__, 127, 21), Foo_4703, *[], **kwargs_4704)

# Processing the call keyword arguments (line 127)
kwargs_4706 = {}
# Getting the type of 'simple_if_idiom_attr' (line 127)
simple_if_idiom_attr_4702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'simple_if_idiom_attr', False)
# Calling simple_if_idiom_attr(args, kwargs) (line 127)
simple_if_idiom_attr_call_result_4707 = invoke(stypy.reporting.localization.Localization(__file__, 127, 0), simple_if_idiom_attr_4702, *[Foo_call_result_4705], **kwargs_4706)


# Call to simple_if_idiom_attr_b(...): (line 128)
# Processing the call arguments (line 128)

# Call to Foo(...): (line 128)
# Processing the call keyword arguments (line 128)
kwargs_4710 = {}
# Getting the type of 'Foo' (line 128)
Foo_4709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 23), 'Foo', False)
# Calling Foo(args, kwargs) (line 128)
Foo_call_result_4711 = invoke(stypy.reporting.localization.Localization(__file__, 128, 23), Foo_4709, *[], **kwargs_4710)

# Processing the call keyword arguments (line 128)
kwargs_4712 = {}
# Getting the type of 'simple_if_idiom_attr_b' (line 128)
simple_if_idiom_attr_b_4708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), 'simple_if_idiom_attr_b', False)
# Calling simple_if_idiom_attr_b(args, kwargs) (line 128)
simple_if_idiom_attr_b_call_result_4713 = invoke(stypy.reporting.localization.Localization(__file__, 128, 0), simple_if_idiom_attr_b_4708, *[Foo_call_result_4711], **kwargs_4712)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
