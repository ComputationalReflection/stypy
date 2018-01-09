
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
11:     if not type(a) == int:
12:         r = a / 3
13:         r2 = a[0]
14:         b = 3
15:     r3 = a / 3
16:     r4 = b / 3
17: 
18: 
19: def simple_if_variant2(a):
20:     b = "hi"
21:     if not a.__class__ is int:
22:         r = a / 3
23:         r2 = a[0]
24:         b = 3
25:     r3 = a / 3
26:     r4 = b / 3
27: 
28: 
29: def simple_if_variant3(a):
30:     b = "hi"
31:     if not a.__class__ == int:
32:         r = a / 3
33:         r2 = a[0]
34:         b = 3
35:     r3 = a / 3
36:     r4 = b / 3
37: 
38: 
39: def simple_if_variant4(a):
40:     b = "hi"
41:     if not int is type(a):
42:         r = a / 3
43:         r2 = a[0]
44:         b = 3
45:     r3 = a / 3
46:     r4 = b / 3
47: 
48: 
49: def simple_if_variant5(a):
50:     b = "hi"
51:     if not int == type(a):
52:         r = a / 3
53:         r2 = a[0]
54:         b = 3
55:     r3 = a / 3
56:     r4 = b / 3
57: 
58: 
59: def simple_if_variant6(a):
60:     b = "hi"
61:     if not int is a.__class__:
62:         r = a / 3
63:         r2 = a[0]
64:         b = 3
65:     r3 = a / 3
66:     r4 = b / 3
67: 
68: 
69: def simple_if_variant7(a):
70:     b = "hi"
71:     if not int == a.__class__:
72:         r = a / 3
73:         r2 = a[0]
74:         b = 3
75:     r3 = a / 3
76:     r4 = b / 3
77: 
78: 
79: def simple_if_variant8(a):
80:     b = "hi"
81:     if type(a) is not int:
82:         r = a / 3
83:         r2 = a[0]
84:         b = 3
85:     r3 = a / 3
86:     r4 = b / 3
87: 
88: simple_if_variant1(union)
89: simple_if_variant2(union)
90: simple_if_variant3(union)
91: simple_if_variant4(union)
92: simple_if_variant5(union)
93: simple_if_variant6(union)
94: simple_if_variant7(union)
95: simple_if_variant8(union)
96: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 1):
int_4242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 9), 'int')
# Assigning a type to the variable 'theInt' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'theInt', int_4242)

# Assigning a Str to a Name (line 2):
str_4243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 9), 'str', 'hi')
# Assigning a type to the variable 'theStr' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'theStr', str_4243)

# Getting the type of 'True' (line 3)
True_4244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 3), 'True')
# Testing the type of an if condition (line 3)
if_condition_4245 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 3, 0), True_4244)
# Assigning a type to the variable 'if_condition_4245' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'if_condition_4245', if_condition_4245)
# SSA begins for if statement (line 3)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Num to a Name (line 4):
int_4246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 12), 'int')
# Assigning a type to the variable 'union' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'union', int_4246)
# SSA branch for the else part of an if statement (line 3)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 6):
str_4247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'str', 'hi')
# Assigning a type to the variable 'union' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'union', str_4247)
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
    str_4248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'b', str_4248)
    
    # Type idiom detected: calculating its left and rigth part (line 11)
    # Getting the type of 'a' (line 11)
    a_4249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'a')
    # Getting the type of 'int' (line 11)
    int_4250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 22), 'int')
    
    (may_be_4251, more_types_in_union_4252) = may_not_be_type(a_4249, int_4250)

    if may_be_4251:

        if more_types_in_union_4252:
            # Runtime conditional SSA (line 11)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 11)
        a_4253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'a')
        # Assigning a type to the variable 'a' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'a', remove_type_from_union(a_4253, int_4250))
        
        # Assigning a BinOp to a Name (line 12):
        # Getting the type of 'a' (line 12)
        a_4254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'a')
        int_4255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 16), 'int')
        # Applying the binary operator 'div' (line 12)
        result_div_4256 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 12), 'div', a_4254, int_4255)
        
        # Assigning a type to the variable 'r' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'r', result_div_4256)
        
        # Assigning a Subscript to a Name (line 13):
        
        # Obtaining the type of the subscript
        int_4257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'int')
        # Getting the type of 'a' (line 13)
        a_4258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 13)
        getitem___4259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 13), a_4258, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 13)
        subscript_call_result_4260 = invoke(stypy.reporting.localization.Localization(__file__, 13, 13), getitem___4259, int_4257)
        
        # Assigning a type to the variable 'r2' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'r2', subscript_call_result_4260)
        
        # Assigning a Num to a Name (line 14):
        int_4261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'int')
        # Assigning a type to the variable 'b' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'b', int_4261)

        if more_types_in_union_4252:
            # SSA join for if statement (line 11)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 15):
    # Getting the type of 'a' (line 15)
    a_4262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 9), 'a')
    int_4263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 13), 'int')
    # Applying the binary operator 'div' (line 15)
    result_div_4264 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 9), 'div', a_4262, int_4263)
    
    # Assigning a type to the variable 'r3' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'r3', result_div_4264)
    
    # Assigning a BinOp to a Name (line 16):
    # Getting the type of 'b' (line 16)
    b_4265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 9), 'b')
    int_4266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'int')
    # Applying the binary operator 'div' (line 16)
    result_div_4267 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 9), 'div', b_4265, int_4266)
    
    # Assigning a type to the variable 'r4' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'r4', result_div_4267)
    
    # ################# End of 'simple_if_variant1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_variant1' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_4268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4268)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_variant1'
    return stypy_return_type_4268

# Assigning a type to the variable 'simple_if_variant1' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'simple_if_variant1', simple_if_variant1)

@norecursion
def simple_if_variant2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_variant2'
    module_type_store = module_type_store.open_function_context('simple_if_variant2', 19, 0, False)
    
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

    
    # Assigning a Str to a Name (line 20):
    str_4269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'b', str_4269)
    
    # Type idiom detected: calculating its left and rigth part (line 21)
    # Getting the type of 'a' (line 21)
    a_4270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'a')
    # Getting the type of 'int' (line 21)
    int_4271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 26), 'int')
    
    (may_be_4272, more_types_in_union_4273) = may_not_be_type(a_4270, int_4271)

    if may_be_4272:

        if more_types_in_union_4273:
            # Runtime conditional SSA (line 21)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 21)
        a_4274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'a')
        # Assigning a type to the variable 'a' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'a', remove_type_from_union(a_4274, int_4271))
        
        # Assigning a BinOp to a Name (line 22):
        # Getting the type of 'a' (line 22)
        a_4275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'a')
        int_4276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'int')
        # Applying the binary operator 'div' (line 22)
        result_div_4277 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 12), 'div', a_4275, int_4276)
        
        # Assigning a type to the variable 'r' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'r', result_div_4277)
        
        # Assigning a Subscript to a Name (line 23):
        
        # Obtaining the type of the subscript
        int_4278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'int')
        # Getting the type of 'a' (line 23)
        a_4279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 23)
        getitem___4280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 13), a_4279, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 23)
        subscript_call_result_4281 = invoke(stypy.reporting.localization.Localization(__file__, 23, 13), getitem___4280, int_4278)
        
        # Assigning a type to the variable 'r2' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'r2', subscript_call_result_4281)
        
        # Assigning a Num to a Name (line 24):
        int_4282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 12), 'int')
        # Assigning a type to the variable 'b' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'b', int_4282)

        if more_types_in_union_4273:
            # SSA join for if statement (line 21)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 25):
    # Getting the type of 'a' (line 25)
    a_4283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 9), 'a')
    int_4284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'int')
    # Applying the binary operator 'div' (line 25)
    result_div_4285 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 9), 'div', a_4283, int_4284)
    
    # Assigning a type to the variable 'r3' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'r3', result_div_4285)
    
    # Assigning a BinOp to a Name (line 26):
    # Getting the type of 'b' (line 26)
    b_4286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 9), 'b')
    int_4287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 13), 'int')
    # Applying the binary operator 'div' (line 26)
    result_div_4288 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 9), 'div', b_4286, int_4287)
    
    # Assigning a type to the variable 'r4' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'r4', result_div_4288)
    
    # ################# End of 'simple_if_variant2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_variant2' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_4289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4289)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_variant2'
    return stypy_return_type_4289

# Assigning a type to the variable 'simple_if_variant2' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'simple_if_variant2', simple_if_variant2)

@norecursion
def simple_if_variant3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_variant3'
    module_type_store = module_type_store.open_function_context('simple_if_variant3', 29, 0, False)
    
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

    
    # Assigning a Str to a Name (line 30):
    str_4290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'b', str_4290)
    
    # Type idiom detected: calculating its left and rigth part (line 31)
    # Getting the type of 'a' (line 31)
    a_4291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'a')
    # Getting the type of 'int' (line 31)
    int_4292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 26), 'int')
    
    (may_be_4293, more_types_in_union_4294) = may_not_be_type(a_4291, int_4292)

    if may_be_4293:

        if more_types_in_union_4294:
            # Runtime conditional SSA (line 31)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 31)
        a_4295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'a')
        # Assigning a type to the variable 'a' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'a', remove_type_from_union(a_4295, int_4292))
        
        # Assigning a BinOp to a Name (line 32):
        # Getting the type of 'a' (line 32)
        a_4296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'a')
        int_4297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 16), 'int')
        # Applying the binary operator 'div' (line 32)
        result_div_4298 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 12), 'div', a_4296, int_4297)
        
        # Assigning a type to the variable 'r' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'r', result_div_4298)
        
        # Assigning a Subscript to a Name (line 33):
        
        # Obtaining the type of the subscript
        int_4299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 15), 'int')
        # Getting the type of 'a' (line 33)
        a_4300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 33)
        getitem___4301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 13), a_4300, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 33)
        subscript_call_result_4302 = invoke(stypy.reporting.localization.Localization(__file__, 33, 13), getitem___4301, int_4299)
        
        # Assigning a type to the variable 'r2' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'r2', subscript_call_result_4302)
        
        # Assigning a Num to a Name (line 34):
        int_4303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 12), 'int')
        # Assigning a type to the variable 'b' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'b', int_4303)

        if more_types_in_union_4294:
            # SSA join for if statement (line 31)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 35):
    # Getting the type of 'a' (line 35)
    a_4304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 9), 'a')
    int_4305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 13), 'int')
    # Applying the binary operator 'div' (line 35)
    result_div_4306 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 9), 'div', a_4304, int_4305)
    
    # Assigning a type to the variable 'r3' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'r3', result_div_4306)
    
    # Assigning a BinOp to a Name (line 36):
    # Getting the type of 'b' (line 36)
    b_4307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 9), 'b')
    int_4308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 13), 'int')
    # Applying the binary operator 'div' (line 36)
    result_div_4309 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 9), 'div', b_4307, int_4308)
    
    # Assigning a type to the variable 'r4' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'r4', result_div_4309)
    
    # ################# End of 'simple_if_variant3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_variant3' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_4310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4310)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_variant3'
    return stypy_return_type_4310

# Assigning a type to the variable 'simple_if_variant3' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'simple_if_variant3', simple_if_variant3)

@norecursion
def simple_if_variant4(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_variant4'
    module_type_store = module_type_store.open_function_context('simple_if_variant4', 39, 0, False)
    
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

    
    # Assigning a Str to a Name (line 40):
    str_4311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'b', str_4311)
    
    # Type idiom detected: calculating its left and rigth part (line 41)
    # Getting the type of 'a' (line 41)
    a_4312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 23), 'a')
    # Getting the type of 'int' (line 41)
    int_4313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'int')
    
    (may_be_4314, more_types_in_union_4315) = may_not_be_type(a_4312, int_4313)

    if may_be_4314:

        if more_types_in_union_4315:
            # Runtime conditional SSA (line 41)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 41)
        a_4316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'a')
        # Assigning a type to the variable 'a' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'a', remove_type_from_union(a_4316, int_4313))
        
        # Assigning a BinOp to a Name (line 42):
        # Getting the type of 'a' (line 42)
        a_4317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'a')
        int_4318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 16), 'int')
        # Applying the binary operator 'div' (line 42)
        result_div_4319 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 12), 'div', a_4317, int_4318)
        
        # Assigning a type to the variable 'r' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'r', result_div_4319)
        
        # Assigning a Subscript to a Name (line 43):
        
        # Obtaining the type of the subscript
        int_4320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 15), 'int')
        # Getting the type of 'a' (line 43)
        a_4321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 43)
        getitem___4322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 13), a_4321, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 43)
        subscript_call_result_4323 = invoke(stypy.reporting.localization.Localization(__file__, 43, 13), getitem___4322, int_4320)
        
        # Assigning a type to the variable 'r2' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'r2', subscript_call_result_4323)
        
        # Assigning a Num to a Name (line 44):
        int_4324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 12), 'int')
        # Assigning a type to the variable 'b' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'b', int_4324)

        if more_types_in_union_4315:
            # SSA join for if statement (line 41)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 45):
    # Getting the type of 'a' (line 45)
    a_4325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 9), 'a')
    int_4326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 13), 'int')
    # Applying the binary operator 'div' (line 45)
    result_div_4327 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 9), 'div', a_4325, int_4326)
    
    # Assigning a type to the variable 'r3' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'r3', result_div_4327)
    
    # Assigning a BinOp to a Name (line 46):
    # Getting the type of 'b' (line 46)
    b_4328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 9), 'b')
    int_4329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 13), 'int')
    # Applying the binary operator 'div' (line 46)
    result_div_4330 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 9), 'div', b_4328, int_4329)
    
    # Assigning a type to the variable 'r4' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'r4', result_div_4330)
    
    # ################# End of 'simple_if_variant4(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_variant4' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_4331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4331)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_variant4'
    return stypy_return_type_4331

# Assigning a type to the variable 'simple_if_variant4' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'simple_if_variant4', simple_if_variant4)

@norecursion
def simple_if_variant5(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_variant5'
    module_type_store = module_type_store.open_function_context('simple_if_variant5', 49, 0, False)
    
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

    
    # Assigning a Str to a Name (line 50):
    str_4332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'b', str_4332)
    
    # Type idiom detected: calculating its left and rigth part (line 51)
    # Getting the type of 'a' (line 51)
    a_4333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 23), 'a')
    # Getting the type of 'int' (line 51)
    int_4334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 11), 'int')
    
    (may_be_4335, more_types_in_union_4336) = may_not_be_type(a_4333, int_4334)

    if may_be_4335:

        if more_types_in_union_4336:
            # Runtime conditional SSA (line 51)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 51)
        a_4337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'a')
        # Assigning a type to the variable 'a' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'a', remove_type_from_union(a_4337, int_4334))
        
        # Assigning a BinOp to a Name (line 52):
        # Getting the type of 'a' (line 52)
        a_4338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'a')
        int_4339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 16), 'int')
        # Applying the binary operator 'div' (line 52)
        result_div_4340 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 12), 'div', a_4338, int_4339)
        
        # Assigning a type to the variable 'r' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'r', result_div_4340)
        
        # Assigning a Subscript to a Name (line 53):
        
        # Obtaining the type of the subscript
        int_4341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 15), 'int')
        # Getting the type of 'a' (line 53)
        a_4342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 53)
        getitem___4343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 13), a_4342, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 53)
        subscript_call_result_4344 = invoke(stypy.reporting.localization.Localization(__file__, 53, 13), getitem___4343, int_4341)
        
        # Assigning a type to the variable 'r2' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'r2', subscript_call_result_4344)
        
        # Assigning a Num to a Name (line 54):
        int_4345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 12), 'int')
        # Assigning a type to the variable 'b' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'b', int_4345)

        if more_types_in_union_4336:
            # SSA join for if statement (line 51)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 55):
    # Getting the type of 'a' (line 55)
    a_4346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 9), 'a')
    int_4347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 13), 'int')
    # Applying the binary operator 'div' (line 55)
    result_div_4348 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 9), 'div', a_4346, int_4347)
    
    # Assigning a type to the variable 'r3' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'r3', result_div_4348)
    
    # Assigning a BinOp to a Name (line 56):
    # Getting the type of 'b' (line 56)
    b_4349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 9), 'b')
    int_4350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 13), 'int')
    # Applying the binary operator 'div' (line 56)
    result_div_4351 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 9), 'div', b_4349, int_4350)
    
    # Assigning a type to the variable 'r4' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'r4', result_div_4351)
    
    # ################# End of 'simple_if_variant5(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_variant5' in the type store
    # Getting the type of 'stypy_return_type' (line 49)
    stypy_return_type_4352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4352)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_variant5'
    return stypy_return_type_4352

# Assigning a type to the variable 'simple_if_variant5' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'simple_if_variant5', simple_if_variant5)

@norecursion
def simple_if_variant6(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_variant6'
    module_type_store = module_type_store.open_function_context('simple_if_variant6', 59, 0, False)
    
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

    
    # Assigning a Str to a Name (line 60):
    str_4353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'b', str_4353)
    
    # Type idiom detected: calculating its left and rigth part (line 61)
    # Getting the type of 'a' (line 61)
    a_4354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'a')
    # Getting the type of 'int' (line 61)
    int_4355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'int')
    
    (may_be_4356, more_types_in_union_4357) = may_not_be_type(a_4354, int_4355)

    if may_be_4356:

        if more_types_in_union_4357:
            # Runtime conditional SSA (line 61)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 61)
        a_4358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'a')
        # Assigning a type to the variable 'a' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'a', remove_type_from_union(a_4358, int_4355))
        
        # Assigning a BinOp to a Name (line 62):
        # Getting the type of 'a' (line 62)
        a_4359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'a')
        int_4360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 16), 'int')
        # Applying the binary operator 'div' (line 62)
        result_div_4361 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 12), 'div', a_4359, int_4360)
        
        # Assigning a type to the variable 'r' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'r', result_div_4361)
        
        # Assigning a Subscript to a Name (line 63):
        
        # Obtaining the type of the subscript
        int_4362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 15), 'int')
        # Getting the type of 'a' (line 63)
        a_4363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___4364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 13), a_4363, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_4365 = invoke(stypy.reporting.localization.Localization(__file__, 63, 13), getitem___4364, int_4362)
        
        # Assigning a type to the variable 'r2' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'r2', subscript_call_result_4365)
        
        # Assigning a Num to a Name (line 64):
        int_4366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 12), 'int')
        # Assigning a type to the variable 'b' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'b', int_4366)

        if more_types_in_union_4357:
            # SSA join for if statement (line 61)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 65):
    # Getting the type of 'a' (line 65)
    a_4367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 9), 'a')
    int_4368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 13), 'int')
    # Applying the binary operator 'div' (line 65)
    result_div_4369 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 9), 'div', a_4367, int_4368)
    
    # Assigning a type to the variable 'r3' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'r3', result_div_4369)
    
    # Assigning a BinOp to a Name (line 66):
    # Getting the type of 'b' (line 66)
    b_4370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 9), 'b')
    int_4371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 13), 'int')
    # Applying the binary operator 'div' (line 66)
    result_div_4372 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 9), 'div', b_4370, int_4371)
    
    # Assigning a type to the variable 'r4' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'r4', result_div_4372)
    
    # ################# End of 'simple_if_variant6(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_variant6' in the type store
    # Getting the type of 'stypy_return_type' (line 59)
    stypy_return_type_4373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4373)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_variant6'
    return stypy_return_type_4373

# Assigning a type to the variable 'simple_if_variant6' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'simple_if_variant6', simple_if_variant6)

@norecursion
def simple_if_variant7(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_variant7'
    module_type_store = module_type_store.open_function_context('simple_if_variant7', 69, 0, False)
    
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

    
    # Assigning a Str to a Name (line 70):
    str_4374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'b', str_4374)
    
    # Type idiom detected: calculating its left and rigth part (line 71)
    # Getting the type of 'a' (line 71)
    a_4375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 18), 'a')
    # Getting the type of 'int' (line 71)
    int_4376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'int')
    
    (may_be_4377, more_types_in_union_4378) = may_not_be_type(a_4375, int_4376)

    if may_be_4377:

        if more_types_in_union_4378:
            # Runtime conditional SSA (line 71)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 71)
        a_4379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'a')
        # Assigning a type to the variable 'a' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'a', remove_type_from_union(a_4379, int_4376))
        
        # Assigning a BinOp to a Name (line 72):
        # Getting the type of 'a' (line 72)
        a_4380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'a')
        int_4381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 16), 'int')
        # Applying the binary operator 'div' (line 72)
        result_div_4382 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 12), 'div', a_4380, int_4381)
        
        # Assigning a type to the variable 'r' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'r', result_div_4382)
        
        # Assigning a Subscript to a Name (line 73):
        
        # Obtaining the type of the subscript
        int_4383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 15), 'int')
        # Getting the type of 'a' (line 73)
        a_4384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 73)
        getitem___4385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 13), a_4384, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
        subscript_call_result_4386 = invoke(stypy.reporting.localization.Localization(__file__, 73, 13), getitem___4385, int_4383)
        
        # Assigning a type to the variable 'r2' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'r2', subscript_call_result_4386)
        
        # Assigning a Num to a Name (line 74):
        int_4387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 12), 'int')
        # Assigning a type to the variable 'b' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'b', int_4387)

        if more_types_in_union_4378:
            # SSA join for if statement (line 71)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 75):
    # Getting the type of 'a' (line 75)
    a_4388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 9), 'a')
    int_4389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 13), 'int')
    # Applying the binary operator 'div' (line 75)
    result_div_4390 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 9), 'div', a_4388, int_4389)
    
    # Assigning a type to the variable 'r3' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'r3', result_div_4390)
    
    # Assigning a BinOp to a Name (line 76):
    # Getting the type of 'b' (line 76)
    b_4391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 9), 'b')
    int_4392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 13), 'int')
    # Applying the binary operator 'div' (line 76)
    result_div_4393 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 9), 'div', b_4391, int_4392)
    
    # Assigning a type to the variable 'r4' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'r4', result_div_4393)
    
    # ################# End of 'simple_if_variant7(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_variant7' in the type store
    # Getting the type of 'stypy_return_type' (line 69)
    stypy_return_type_4394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4394)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_variant7'
    return stypy_return_type_4394

# Assigning a type to the variable 'simple_if_variant7' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'simple_if_variant7', simple_if_variant7)

@norecursion
def simple_if_variant8(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_variant8'
    module_type_store = module_type_store.open_function_context('simple_if_variant8', 79, 0, False)
    
    # Passed parameters checking function
    simple_if_variant8.stypy_localization = localization
    simple_if_variant8.stypy_type_of_self = None
    simple_if_variant8.stypy_type_store = module_type_store
    simple_if_variant8.stypy_function_name = 'simple_if_variant8'
    simple_if_variant8.stypy_param_names_list = ['a']
    simple_if_variant8.stypy_varargs_param_name = None
    simple_if_variant8.stypy_kwargs_param_name = None
    simple_if_variant8.stypy_call_defaults = defaults
    simple_if_variant8.stypy_call_varargs = varargs
    simple_if_variant8.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_variant8', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_variant8', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_variant8(...)' code ##################

    
    # Assigning a Str to a Name (line 80):
    str_4395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'b', str_4395)
    
    # Type idiom detected: calculating its left and rigth part (line 81)
    # Getting the type of 'a' (line 81)
    a_4396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'a')
    # Getting the type of 'int' (line 81)
    int_4397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'int')
    
    (may_be_4398, more_types_in_union_4399) = may_not_be_type(a_4396, int_4397)

    if may_be_4398:

        if more_types_in_union_4399:
            # Runtime conditional SSA (line 81)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 81)
        a_4400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'a')
        # Assigning a type to the variable 'a' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'a', remove_type_from_union(a_4400, int_4397))
        
        # Assigning a BinOp to a Name (line 82):
        # Getting the type of 'a' (line 82)
        a_4401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'a')
        int_4402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 16), 'int')
        # Applying the binary operator 'div' (line 82)
        result_div_4403 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 12), 'div', a_4401, int_4402)
        
        # Assigning a type to the variable 'r' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'r', result_div_4403)
        
        # Assigning a Subscript to a Name (line 83):
        
        # Obtaining the type of the subscript
        int_4404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 15), 'int')
        # Getting the type of 'a' (line 83)
        a_4405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___4406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 13), a_4405, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_4407 = invoke(stypy.reporting.localization.Localization(__file__, 83, 13), getitem___4406, int_4404)
        
        # Assigning a type to the variable 'r2' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'r2', subscript_call_result_4407)
        
        # Assigning a Num to a Name (line 84):
        int_4408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 12), 'int')
        # Assigning a type to the variable 'b' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'b', int_4408)

        if more_types_in_union_4399:
            # SSA join for if statement (line 81)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 85):
    # Getting the type of 'a' (line 85)
    a_4409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 9), 'a')
    int_4410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 13), 'int')
    # Applying the binary operator 'div' (line 85)
    result_div_4411 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 9), 'div', a_4409, int_4410)
    
    # Assigning a type to the variable 'r3' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'r3', result_div_4411)
    
    # Assigning a BinOp to a Name (line 86):
    # Getting the type of 'b' (line 86)
    b_4412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 9), 'b')
    int_4413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 13), 'int')
    # Applying the binary operator 'div' (line 86)
    result_div_4414 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 9), 'div', b_4412, int_4413)
    
    # Assigning a type to the variable 'r4' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'r4', result_div_4414)
    
    # ################# End of 'simple_if_variant8(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_variant8' in the type store
    # Getting the type of 'stypy_return_type' (line 79)
    stypy_return_type_4415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4415)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_variant8'
    return stypy_return_type_4415

# Assigning a type to the variable 'simple_if_variant8' (line 79)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'simple_if_variant8', simple_if_variant8)

# Call to simple_if_variant1(...): (line 88)
# Processing the call arguments (line 88)
# Getting the type of 'union' (line 88)
union_4417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'union', False)
# Processing the call keyword arguments (line 88)
kwargs_4418 = {}
# Getting the type of 'simple_if_variant1' (line 88)
simple_if_variant1_4416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'simple_if_variant1', False)
# Calling simple_if_variant1(args, kwargs) (line 88)
simple_if_variant1_call_result_4419 = invoke(stypy.reporting.localization.Localization(__file__, 88, 0), simple_if_variant1_4416, *[union_4417], **kwargs_4418)


# Call to simple_if_variant2(...): (line 89)
# Processing the call arguments (line 89)
# Getting the type of 'union' (line 89)
union_4421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 19), 'union', False)
# Processing the call keyword arguments (line 89)
kwargs_4422 = {}
# Getting the type of 'simple_if_variant2' (line 89)
simple_if_variant2_4420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'simple_if_variant2', False)
# Calling simple_if_variant2(args, kwargs) (line 89)
simple_if_variant2_call_result_4423 = invoke(stypy.reporting.localization.Localization(__file__, 89, 0), simple_if_variant2_4420, *[union_4421], **kwargs_4422)


# Call to simple_if_variant3(...): (line 90)
# Processing the call arguments (line 90)
# Getting the type of 'union' (line 90)
union_4425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'union', False)
# Processing the call keyword arguments (line 90)
kwargs_4426 = {}
# Getting the type of 'simple_if_variant3' (line 90)
simple_if_variant3_4424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'simple_if_variant3', False)
# Calling simple_if_variant3(args, kwargs) (line 90)
simple_if_variant3_call_result_4427 = invoke(stypy.reporting.localization.Localization(__file__, 90, 0), simple_if_variant3_4424, *[union_4425], **kwargs_4426)


# Call to simple_if_variant4(...): (line 91)
# Processing the call arguments (line 91)
# Getting the type of 'union' (line 91)
union_4429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 19), 'union', False)
# Processing the call keyword arguments (line 91)
kwargs_4430 = {}
# Getting the type of 'simple_if_variant4' (line 91)
simple_if_variant4_4428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'simple_if_variant4', False)
# Calling simple_if_variant4(args, kwargs) (line 91)
simple_if_variant4_call_result_4431 = invoke(stypy.reporting.localization.Localization(__file__, 91, 0), simple_if_variant4_4428, *[union_4429], **kwargs_4430)


# Call to simple_if_variant5(...): (line 92)
# Processing the call arguments (line 92)
# Getting the type of 'union' (line 92)
union_4433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 19), 'union', False)
# Processing the call keyword arguments (line 92)
kwargs_4434 = {}
# Getting the type of 'simple_if_variant5' (line 92)
simple_if_variant5_4432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'simple_if_variant5', False)
# Calling simple_if_variant5(args, kwargs) (line 92)
simple_if_variant5_call_result_4435 = invoke(stypy.reporting.localization.Localization(__file__, 92, 0), simple_if_variant5_4432, *[union_4433], **kwargs_4434)


# Call to simple_if_variant6(...): (line 93)
# Processing the call arguments (line 93)
# Getting the type of 'union' (line 93)
union_4437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 19), 'union', False)
# Processing the call keyword arguments (line 93)
kwargs_4438 = {}
# Getting the type of 'simple_if_variant6' (line 93)
simple_if_variant6_4436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'simple_if_variant6', False)
# Calling simple_if_variant6(args, kwargs) (line 93)
simple_if_variant6_call_result_4439 = invoke(stypy.reporting.localization.Localization(__file__, 93, 0), simple_if_variant6_4436, *[union_4437], **kwargs_4438)


# Call to simple_if_variant7(...): (line 94)
# Processing the call arguments (line 94)
# Getting the type of 'union' (line 94)
union_4441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 19), 'union', False)
# Processing the call keyword arguments (line 94)
kwargs_4442 = {}
# Getting the type of 'simple_if_variant7' (line 94)
simple_if_variant7_4440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'simple_if_variant7', False)
# Calling simple_if_variant7(args, kwargs) (line 94)
simple_if_variant7_call_result_4443 = invoke(stypy.reporting.localization.Localization(__file__, 94, 0), simple_if_variant7_4440, *[union_4441], **kwargs_4442)


# Call to simple_if_variant8(...): (line 95)
# Processing the call arguments (line 95)
# Getting the type of 'union' (line 95)
union_4445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 19), 'union', False)
# Processing the call keyword arguments (line 95)
kwargs_4446 = {}
# Getting the type of 'simple_if_variant8' (line 95)
simple_if_variant8_4444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'simple_if_variant8', False)
# Calling simple_if_variant8(args, kwargs) (line 95)
simple_if_variant8_call_result_4447 = invoke(stypy.reporting.localization.Localization(__file__, 95, 0), simple_if_variant8_4444, *[union_4445], **kwargs_4446)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
