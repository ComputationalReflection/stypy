
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
11: 
12:     if '__div__' in a.__class__.__dict__:
13:         r = a / 3
14:         r2 = a[0]
15:         b = 3
16:     r3 = a / 3
17:     r4 = b / 3
18: 
19: 
20: def simple_if_hasattr_variant2(a):
21:     b = "hi"
22:     if type(a).__dict__.has_key('__div__'):
23:         r = a / 3
24:         r2 = a[0]
25:         b = 3
26:     r3 = a / 3
27:     r4 = b / 3
28: 
29: 
30: def simple_if_hasattr_variant3(a):
31:     b = "hi"
32:     if '__div__' in type(a).__dict__:
33:         r = a / 3
34:         r2 = a[0]
35:         b = 3
36:     r3 = a / 3
37:     r4 = b / 3
38: 
39: 
40: def simple_if_hasattr_variant4(a):
41:     b = "hi"
42:     if '__div__' in dir(type(a)):
43:         r = a / 3
44:         r2 = a[0]
45:         b = 3
46:     r3 = a / 3
47:     r4 = b / 3
48: 
49: 
50: simple_if_hasattr_variant1(union)
51: simple_if_hasattr_variant2(union)
52: simple_if_hasattr_variant3(union)
53: simple_if_hasattr_variant4(union)

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 1):
int_3525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 9), 'int')
# Assigning a type to the variable 'theInt' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'theInt', int_3525)

# Assigning a Str to a Name (line 2):
str_3526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 9), 'str', 'hi')
# Assigning a type to the variable 'theStr' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'theStr', str_3526)

# Getting the type of 'True' (line 3)
True_3527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 3), 'True')
# Testing the type of an if condition (line 3)
if_condition_3528 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 3, 0), True_3527)
# Assigning a type to the variable 'if_condition_3528' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'if_condition_3528', if_condition_3528)
# SSA begins for if statement (line 3)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Num to a Name (line 4):
int_3529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 12), 'int')
# Assigning a type to the variable 'union' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'union', int_3529)
# SSA branch for the else part of an if statement (line 3)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 6):
str_3530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'str', 'hi')
# Assigning a type to the variable 'union' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'union', str_3530)
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
    str_3531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'b', str_3531)
    
    # Type idiom detected: calculating its left and rigth part (line 12)
    str_3532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 7), 'str', '__div__')
    # Getting the type of 'a' (line 12)
    a_3533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 20), 'a')
    
    (may_be_3534, more_types_in_union_3535) = may_provide_member(str_3532, a_3533)

    if may_be_3534:

        if more_types_in_union_3535:
            # Runtime conditional SSA (line 12)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'a', remove_not_member_provider_from_union(a_3533, '__div__'))
        
        # Assigning a BinOp to a Name (line 13):
        # Getting the type of 'a' (line 13)
        a_3536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'a')
        int_3537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 16), 'int')
        # Applying the binary operator 'div' (line 13)
        result_div_3538 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 12), 'div', a_3536, int_3537)
        
        # Assigning a type to the variable 'r' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'r', result_div_3538)
        
        # Assigning a Subscript to a Name (line 14):
        
        # Obtaining the type of the subscript
        int_3539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'int')
        # Getting the type of 'a' (line 14)
        a_3540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 14)
        getitem___3541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 13), a_3540, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 14)
        subscript_call_result_3542 = invoke(stypy.reporting.localization.Localization(__file__, 14, 13), getitem___3541, int_3539)
        
        # Assigning a type to the variable 'r2' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'r2', subscript_call_result_3542)
        
        # Assigning a Num to a Name (line 15):
        int_3543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 12), 'int')
        # Assigning a type to the variable 'b' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'b', int_3543)

        if more_types_in_union_3535:
            # SSA join for if statement (line 12)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 16):
    # Getting the type of 'a' (line 16)
    a_3544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 9), 'a')
    int_3545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'int')
    # Applying the binary operator 'div' (line 16)
    result_div_3546 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 9), 'div', a_3544, int_3545)
    
    # Assigning a type to the variable 'r3' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'r3', result_div_3546)
    
    # Assigning a BinOp to a Name (line 17):
    # Getting the type of 'b' (line 17)
    b_3547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 9), 'b')
    int_3548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 13), 'int')
    # Applying the binary operator 'div' (line 17)
    result_div_3549 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 9), 'div', b_3547, int_3548)
    
    # Assigning a type to the variable 'r4' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'r4', result_div_3549)
    
    # ################# End of 'simple_if_hasattr_variant1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_hasattr_variant1' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_3550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3550)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_hasattr_variant1'
    return stypy_return_type_3550

# Assigning a type to the variable 'simple_if_hasattr_variant1' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'simple_if_hasattr_variant1', simple_if_hasattr_variant1)

@norecursion
def simple_if_hasattr_variant2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_hasattr_variant2'
    module_type_store = module_type_store.open_function_context('simple_if_hasattr_variant2', 20, 0, False)
    
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

    
    # Assigning a Str to a Name (line 21):
    str_3551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'b', str_3551)
    
    # Type idiom detected: calculating its left and rigth part (line 22)
    str_3552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 32), 'str', '__div__')
    # Getting the type of 'a' (line 22)
    a_3553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'a')
    
    (may_be_3554, more_types_in_union_3555) = may_provide_member(str_3552, a_3553)

    if may_be_3554:

        if more_types_in_union_3555:
            # Runtime conditional SSA (line 22)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'a', remove_not_member_provider_from_union(a_3553, '__div__'))
        
        # Assigning a BinOp to a Name (line 23):
        # Getting the type of 'a' (line 23)
        a_3556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'a')
        int_3557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 16), 'int')
        # Applying the binary operator 'div' (line 23)
        result_div_3558 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 12), 'div', a_3556, int_3557)
        
        # Assigning a type to the variable 'r' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'r', result_div_3558)
        
        # Assigning a Subscript to a Name (line 24):
        
        # Obtaining the type of the subscript
        int_3559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 15), 'int')
        # Getting the type of 'a' (line 24)
        a_3560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 24)
        getitem___3561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 13), a_3560, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 24)
        subscript_call_result_3562 = invoke(stypy.reporting.localization.Localization(__file__, 24, 13), getitem___3561, int_3559)
        
        # Assigning a type to the variable 'r2' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'r2', subscript_call_result_3562)
        
        # Assigning a Num to a Name (line 25):
        int_3563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 12), 'int')
        # Assigning a type to the variable 'b' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'b', int_3563)

        if more_types_in_union_3555:
            # SSA join for if statement (line 22)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 26):
    # Getting the type of 'a' (line 26)
    a_3564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 9), 'a')
    int_3565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 13), 'int')
    # Applying the binary operator 'div' (line 26)
    result_div_3566 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 9), 'div', a_3564, int_3565)
    
    # Assigning a type to the variable 'r3' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'r3', result_div_3566)
    
    # Assigning a BinOp to a Name (line 27):
    # Getting the type of 'b' (line 27)
    b_3567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 9), 'b')
    int_3568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 13), 'int')
    # Applying the binary operator 'div' (line 27)
    result_div_3569 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 9), 'div', b_3567, int_3568)
    
    # Assigning a type to the variable 'r4' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'r4', result_div_3569)
    
    # ################# End of 'simple_if_hasattr_variant2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_hasattr_variant2' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_3570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3570)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_hasattr_variant2'
    return stypy_return_type_3570

# Assigning a type to the variable 'simple_if_hasattr_variant2' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'simple_if_hasattr_variant2', simple_if_hasattr_variant2)

@norecursion
def simple_if_hasattr_variant3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_hasattr_variant3'
    module_type_store = module_type_store.open_function_context('simple_if_hasattr_variant3', 30, 0, False)
    
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

    
    # Assigning a Str to a Name (line 31):
    str_3571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'b', str_3571)
    
    # Type idiom detected: calculating its left and rigth part (line 32)
    str_3572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 7), 'str', '__div__')
    # Getting the type of 'a' (line 32)
    a_3573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 25), 'a')
    
    (may_be_3574, more_types_in_union_3575) = may_provide_member(str_3572, a_3573)

    if may_be_3574:

        if more_types_in_union_3575:
            # Runtime conditional SSA (line 32)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'a', remove_not_member_provider_from_union(a_3573, '__div__'))
        
        # Assigning a BinOp to a Name (line 33):
        # Getting the type of 'a' (line 33)
        a_3576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'a')
        int_3577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 16), 'int')
        # Applying the binary operator 'div' (line 33)
        result_div_3578 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 12), 'div', a_3576, int_3577)
        
        # Assigning a type to the variable 'r' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'r', result_div_3578)
        
        # Assigning a Subscript to a Name (line 34):
        
        # Obtaining the type of the subscript
        int_3579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 15), 'int')
        # Getting the type of 'a' (line 34)
        a_3580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 34)
        getitem___3581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 13), a_3580, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 34)
        subscript_call_result_3582 = invoke(stypy.reporting.localization.Localization(__file__, 34, 13), getitem___3581, int_3579)
        
        # Assigning a type to the variable 'r2' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'r2', subscript_call_result_3582)
        
        # Assigning a Num to a Name (line 35):
        int_3583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 12), 'int')
        # Assigning a type to the variable 'b' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'b', int_3583)

        if more_types_in_union_3575:
            # SSA join for if statement (line 32)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 36):
    # Getting the type of 'a' (line 36)
    a_3584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 9), 'a')
    int_3585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 13), 'int')
    # Applying the binary operator 'div' (line 36)
    result_div_3586 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 9), 'div', a_3584, int_3585)
    
    # Assigning a type to the variable 'r3' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'r3', result_div_3586)
    
    # Assigning a BinOp to a Name (line 37):
    # Getting the type of 'b' (line 37)
    b_3587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 9), 'b')
    int_3588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 13), 'int')
    # Applying the binary operator 'div' (line 37)
    result_div_3589 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 9), 'div', b_3587, int_3588)
    
    # Assigning a type to the variable 'r4' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'r4', result_div_3589)
    
    # ################# End of 'simple_if_hasattr_variant3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_hasattr_variant3' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_3590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3590)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_hasattr_variant3'
    return stypy_return_type_3590

# Assigning a type to the variable 'simple_if_hasattr_variant3' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'simple_if_hasattr_variant3', simple_if_hasattr_variant3)

@norecursion
def simple_if_hasattr_variant4(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_hasattr_variant4'
    module_type_store = module_type_store.open_function_context('simple_if_hasattr_variant4', 40, 0, False)
    
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

    
    # Assigning a Str to a Name (line 41):
    str_3591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'b', str_3591)
    
    # Type idiom detected: calculating its left and rigth part (line 42)
    str_3592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 7), 'str', '__div__')
    # Getting the type of 'a' (line 42)
    a_3593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 29), 'a')
    
    (may_be_3594, more_types_in_union_3595) = may_provide_member(str_3592, a_3593)

    if may_be_3594:

        if more_types_in_union_3595:
            # Runtime conditional SSA (line 42)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'a', remove_not_member_provider_from_union(a_3593, '__div__'))
        
        # Assigning a BinOp to a Name (line 43):
        # Getting the type of 'a' (line 43)
        a_3596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'a')
        int_3597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 16), 'int')
        # Applying the binary operator 'div' (line 43)
        result_div_3598 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 12), 'div', a_3596, int_3597)
        
        # Assigning a type to the variable 'r' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'r', result_div_3598)
        
        # Assigning a Subscript to a Name (line 44):
        
        # Obtaining the type of the subscript
        int_3599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 15), 'int')
        # Getting the type of 'a' (line 44)
        a_3600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 44)
        getitem___3601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 13), a_3600, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 44)
        subscript_call_result_3602 = invoke(stypy.reporting.localization.Localization(__file__, 44, 13), getitem___3601, int_3599)
        
        # Assigning a type to the variable 'r2' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'r2', subscript_call_result_3602)
        
        # Assigning a Num to a Name (line 45):
        int_3603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 12), 'int')
        # Assigning a type to the variable 'b' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'b', int_3603)

        if more_types_in_union_3595:
            # SSA join for if statement (line 42)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 46):
    # Getting the type of 'a' (line 46)
    a_3604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 9), 'a')
    int_3605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 13), 'int')
    # Applying the binary operator 'div' (line 46)
    result_div_3606 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 9), 'div', a_3604, int_3605)
    
    # Assigning a type to the variable 'r3' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'r3', result_div_3606)
    
    # Assigning a BinOp to a Name (line 47):
    # Getting the type of 'b' (line 47)
    b_3607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 9), 'b')
    int_3608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 13), 'int')
    # Applying the binary operator 'div' (line 47)
    result_div_3609 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 9), 'div', b_3607, int_3608)
    
    # Assigning a type to the variable 'r4' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'r4', result_div_3609)
    
    # ################# End of 'simple_if_hasattr_variant4(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_hasattr_variant4' in the type store
    # Getting the type of 'stypy_return_type' (line 40)
    stypy_return_type_3610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3610)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_hasattr_variant4'
    return stypy_return_type_3610

# Assigning a type to the variable 'simple_if_hasattr_variant4' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'simple_if_hasattr_variant4', simple_if_hasattr_variant4)

# Call to simple_if_hasattr_variant1(...): (line 50)
# Processing the call arguments (line 50)
# Getting the type of 'union' (line 50)
union_3612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 27), 'union', False)
# Processing the call keyword arguments (line 50)
kwargs_3613 = {}
# Getting the type of 'simple_if_hasattr_variant1' (line 50)
simple_if_hasattr_variant1_3611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'simple_if_hasattr_variant1', False)
# Calling simple_if_hasattr_variant1(args, kwargs) (line 50)
simple_if_hasattr_variant1_call_result_3614 = invoke(stypy.reporting.localization.Localization(__file__, 50, 0), simple_if_hasattr_variant1_3611, *[union_3612], **kwargs_3613)


# Call to simple_if_hasattr_variant2(...): (line 51)
# Processing the call arguments (line 51)
# Getting the type of 'union' (line 51)
union_3616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 27), 'union', False)
# Processing the call keyword arguments (line 51)
kwargs_3617 = {}
# Getting the type of 'simple_if_hasattr_variant2' (line 51)
simple_if_hasattr_variant2_3615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'simple_if_hasattr_variant2', False)
# Calling simple_if_hasattr_variant2(args, kwargs) (line 51)
simple_if_hasattr_variant2_call_result_3618 = invoke(stypy.reporting.localization.Localization(__file__, 51, 0), simple_if_hasattr_variant2_3615, *[union_3616], **kwargs_3617)


# Call to simple_if_hasattr_variant3(...): (line 52)
# Processing the call arguments (line 52)
# Getting the type of 'union' (line 52)
union_3620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 27), 'union', False)
# Processing the call keyword arguments (line 52)
kwargs_3621 = {}
# Getting the type of 'simple_if_hasattr_variant3' (line 52)
simple_if_hasattr_variant3_3619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'simple_if_hasattr_variant3', False)
# Calling simple_if_hasattr_variant3(args, kwargs) (line 52)
simple_if_hasattr_variant3_call_result_3622 = invoke(stypy.reporting.localization.Localization(__file__, 52, 0), simple_if_hasattr_variant3_3619, *[union_3620], **kwargs_3621)


# Call to simple_if_hasattr_variant4(...): (line 53)
# Processing the call arguments (line 53)
# Getting the type of 'union' (line 53)
union_3624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 27), 'union', False)
# Processing the call keyword arguments (line 53)
kwargs_3625 = {}
# Getting the type of 'simple_if_hasattr_variant4' (line 53)
simple_if_hasattr_variant4_3623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'simple_if_hasattr_variant4', False)
# Calling simple_if_hasattr_variant4(args, kwargs) (line 53)
simple_if_hasattr_variant4_call_result_3626 = invoke(stypy.reporting.localization.Localization(__file__, 53, 0), simple_if_hasattr_variant4_3623, *[union_3624], **kwargs_3625)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
